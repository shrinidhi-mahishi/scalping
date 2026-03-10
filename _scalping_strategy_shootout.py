"""
Scalping strategy shootout on Nifty 50 (48 stocks) over the last 90 days.

Compared structures:
  1. Current PDL
  2. Pre-armed PDL breakout
  3. PDL retest entry
  4. ORB
  5. Pre-armed PDL + Retest
  6. ORB + PDL Retest

Common framework:
  - 3-minute candles built from cached 1-minute data
  - 09:30 to 13:00 entry window
  - max 2 SL/day/stock
  - 30-minute cooldown
  - forced EOD exit
  - no fees/slippage (strategy-only comparison)
  - RR sweep geared for scalping
"""

from __future__ import annotations

from datetime import time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from fetch_data import load_csv

CASH = 50_000.0
TODAY = pd.Timestamp.now().normalize()
START = TODAY - pd.Timedelta(days=90)
_F = str.maketrans({"&": ""})

RISK_PCT = 0.015
LEV_CAP = 5.0
MAX_SL_PER_DAY = 2
COOLDOWN = timedelta(minutes=30)
ENTRY_START = time(9, 30)
ENTRY_END = time(13, 0)
RRS = [0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50]

NIFTY50 = [
    ("ADANIENT", "Infra"), ("ADANIPORTS", "Ports"), ("APOLLOHOSP", "Healthcare"),
    ("ASIANPAINT", "Paint"), ("AXISBANK", "Banking"), ("BAJAJ-AUTO", "Auto"),
    ("BAJAJFINSV", "Finance"), ("BAJFINANCE", "Finance"), ("BEL", "Defence"),
    ("BHARTIARTL", "Telecom"), ("CIPLA", "Pharma"), ("COALINDIA", "Mining"),
    ("DRREDDY", "Pharma"), ("EICHERMOT", "Auto"), ("GRASIM", "Cement"),
    ("HCLTECH", "IT"), ("HDFCBANK", "Banking"), ("HDFCLIFE", "Finance"),
    ("HINDALCO", "Metals"), ("HINDUNILVR", "FMCG"), ("ICICIBANK", "Banking"),
    ("INDIGO", "Aviation"), ("INFY", "IT"), ("ITC", "FMCG"),
    ("JIOFIN", "Finance"), ("JSWSTEEL", "Steel"), ("KOTAKBANK", "Banking"),
    ("LT", "Infra"), ("M&M", "Auto"), ("MARUTI", "Auto"),
    ("NESTLEIND", "Consumer"), ("NTPC", "Power"), ("ONGC", "Oil"),
    ("POWERGRID", "Power"), ("RELIANCE", "Conglomerate"), ("SBILIFE", "Insurance"),
    ("SBIN", "Banking"), ("SHRIRAMFIN", "Finance"), ("SUNPHARMA", "Pharma"),
    ("TATACONSUM", "Consumer"), ("TATAMOTORS", "Auto"), ("TATASTEEL", "Steel"),
    ("TCS", "IT"), ("TECHM", "IT"), ("TITAN", "Consumer"),
    ("TRENT", "Retail"), ("ULTRACEMCO", "Cement"), ("WIPRO", "IT"),
]


def csv_1min(sym: str) -> Path:
    return Path("data") / f"{sym.translate(_F)}_1min.csv"


def resample_3min(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("3min").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    ).dropna(subset=["open"])


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    tp = (out["high"] + out["low"] + out["close"]) / 3.0
    day = out.index.date
    out["vwap"] = (tp * out["volume"]).groupby(day).cumsum() / out["volume"].groupby(day).cumsum()

    delta = out["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_g = gain.ewm(alpha=1.0 / 9, min_periods=9, adjust=False).mean()
    avg_l = loss.ewm(alpha=1.0 / 9, min_periods=9, adjust=False).mean()
    rs = avg_g / avg_l.replace(0, np.nan)
    out["rsi"] = 100.0 - 100.0 / (1.0 + rs)

    out["vol_sma"] = out["volume"].rolling(20).mean()

    pc = out["close"].shift(1)
    tr = np.maximum(out["high"], pc) - np.minimum(out["low"], pc)
    out["atr"] = tr.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()

    return out


def get_day_blocks(ind: pd.DataFrame) -> list[dict]:
    ind = ind[ind.index >= START]
    days = sorted(ind.index.normalize().unique())
    out = []
    for i in range(1, len(days)):
        td, pd_ = days[i], days[i - 1]
        prev_rows = ind[ind.index.normalize() == pd_]
        today_rows = ind[ind.index.normalize() == td]
        if prev_rows.empty or today_rows.empty:
            continue
        out.append(
            {
                "date": td,
                "prev": prev_rows,
                "today": today_rows,
                "pdl_h": float(prev_rows["high"].max()),
                "pdl_l": float(prev_rows["low"].min()),
            }
        )
    return out


def open_position(sym: str, trade_date, direction: str, entry: float, stop_dist: float, rr: float, atr: float, ts) -> dict:
    qty_risk = int(CASH * RISK_PCT / stop_dist)
    qty_lev = int(CASH * LEV_CAP / entry)
    qty = max(min(qty_risk, qty_lev), 1)
    sl = entry - stop_dist if direction == "LONG" else entry + stop_dist
    tp = entry + stop_dist * rr if direction == "LONG" else entry - stop_dist * rr
    return {
        "sym": sym,
        "date": trade_date,
        "d": direction,
        "e": entry,
        "sl": sl,
        "tp": tp,
        "atr": atr,
        "q": qty,
        "h": ts.time().hour,
        "t": ts.time(),
    }


def check_exit(row, pos) -> tuple[str | None, float]:
    if pos["d"] == "LONG":
        if row["low"] <= pos["sl"]:
            return "SL", (pos["sl"] - pos["e"]) * pos["q"]
        if row["high"] >= pos["tp"]:
            return "TP", (pos["tp"] - pos["e"]) * pos["q"]
    else:
        if row["high"] >= pos["sl"]:
            return "SL", (pos["e"] - pos["sl"]) * pos["q"]
        if row["low"] <= pos["tp"]:
            return "TP", (pos["e"] - pos["tp"]) * pos["q"]
    return None, 0.0


def eod_exit(trades: list[dict], pos: dict | None, last_close: float) -> None:
    if pos is None:
        return
    pnl = (last_close - pos["e"]) * pos["q"] if pos["d"] == "LONG" else (pos["e"] - last_close) * pos["q"]
    trades.append({**pos, "pnl": pnl, "res": "EOD"})


def in_window(ts) -> bool:
    return ENTRY_START <= ts.time() <= ENTRY_END


def analyze(trades: list[dict]) -> dict:
    if not trades:
        return {"n": 0, "wr": 0.0, "pnl": 0.0, "tp": 0, "sl": 0, "eod": 0, "dd": 0.0, "pf": 0.0, "avg": 0.0}
    tdf = pd.DataFrame(trades)
    n = len(tdf)
    pnl = float(tdf["pnl"].sum())
    wins = int((tdf["pnl"] > 0).sum())
    tp = int((tdf["res"] == "TP").sum())
    sl = int((tdf["res"] == "SL").sum())
    eod = int((tdf["res"] == "EOD").sum())
    cumul = tdf["pnl"].cumsum()
    dd = float((cumul - cumul.cummax()).min())
    gw = float(tdf.loc[tdf["pnl"] > 0, "pnl"].sum())
    gl = abs(float(tdf.loc[tdf["pnl"] <= 0, "pnl"].sum()))
    pf = gw / gl if gl > 0 else (9.99 if gw > 0 else 0.0)
    return {"n": n, "wr": wins / n * 100, "pnl": pnl, "tp": tp, "sl": sl, "eod": eod, "dd": dd, "pf": min(pf, 9.99), "avg": pnl / n}


def summarize_strategy(strategy_name: str, rr: float, stock_trades: dict[str, list[dict]], sectors: dict[str, str]) -> dict:
    all_trades = []
    per_stock = {}
    for sym, trades in stock_trades.items():
        all_trades.extend(trades)
        per_stock[sym] = sum(t["pnl"] for t in trades)

    overall = analyze(all_trades)
    ranked = sorted(per_stock.items(), key=lambda x: x[1], reverse=True)
    top10_syms = {sym for sym, _ in ranked[:10]}
    top10_trades = [t for t in all_trades if t["sym"] in top10_syms]
    top10 = analyze(top10_trades)

    dates = sorted({t["date"] for t in top10_trades})
    recent10_dates = set(dates[-10:])
    recent10_trades = [t for t in top10_trades if t["date"] in recent10_dates]
    recent10 = analyze(recent10_trades)

    return {
        "strategy": strategy_name,
        "rr": rr,
        "overall": overall,
        "top10": top10,
        "recent10": recent10,
        "ranked": ranked,
        "sectors": sectors,
    }


def simulate_current_pdl(sym: str, blocks: list[dict], rr: float) -> list[dict]:
    trades = []
    for block in blocks:
        rows = block["today"]
        pdl_h = block["pdl_h"]
        pdl_l = block["pdl_l"]
        pos = None
        cd_until = None
        dsl = 0
        dirs_used = set()
        for i in range(len(rows)):
            row = rows.iloc[i]
            ts = rows.index[i]
            c = float(row["close"])
            prev_close = float(rows.iloc[i - 1]["close"]) if i > 0 else None

            if pos is not None:
                hit, pnl = check_exit(row, pos)
                if hit:
                    trades.append({**pos, "pnl": pnl, "res": hit})
                    if hit == "SL":
                        dsl += 1
                    pos = None
                continue

            if not in_window(ts) or (cd_until and ts < cd_until) or dsl >= MAX_SL_PER_DAY:
                continue

            rsi, atr, vwap, vsma = row["rsi"], row["atr"], row["vwap"], row["vol_sma"]
            if prev_close is None or pd.isna(rsi) or pd.isna(atr) or atr <= 0 or pd.isna(vwap) or pd.isna(vsma):
                continue

            vol_ok = row["volume"] > vsma if vsma > 0 else False
            direction = None
            if vol_ok:
                if "LONG" not in dirs_used and prev_close <= pdl_h and c > pdl_h and c > vwap and rsi > 50:
                    direction = "LONG"
                elif "SHORT" not in dirs_used and prev_close >= pdl_l and c < pdl_l and c < vwap and rsi < 50:
                    direction = "SHORT"

            if direction:
                pos = open_position(sym, block["date"], direction, c, float(atr) * 1.2, rr, float(atr), ts)
                dirs_used.add(direction)
                cd_until = ts + COOLDOWN

        eod_exit(trades, pos, float(rows.iloc[-1]["close"]))
    return trades


def simulate_prearmed_pdl(sym: str, blocks: list[dict], rr: float) -> list[dict]:
    trades = []
    for block in blocks:
        rows = block["today"]
        pdl_h = block["pdl_h"]
        pdl_l = block["pdl_l"]
        pos = None
        cd_until = None
        dsl = 0
        dirs_used = set()
        activate_idx = None

        for i in range(len(rows)):
            row = rows.iloc[i]
            ts = rows.index[i]
            atr = row["atr"]
            if pd.isna(atr) or atr <= 0:
                continue

            if pos is not None:
                if activate_idx is not None and i >= activate_idx:
                    hit, pnl = check_exit(row, pos)
                    if hit:
                        trades.append({**pos, "pnl": pnl, "res": hit})
                        if hit == "SL":
                            dsl += 1
                        pos, activate_idx = None, None
                continue

            if not in_window(ts) or (cd_until and ts < cd_until) or dsl >= MAX_SL_PER_DAY:
                continue

            prev_close = float(rows.iloc[i - 1]["close"]) if i > 0 else None
            if prev_close is None:
                continue

            buffer = float(atr) * 0.10
            direction = None
            entry = None

            if "LONG" not in dirs_used and prev_close <= pdl_h and row["high"] >= (pdl_h + buffer) and row["close"] >= pdl_h:
                direction = "LONG"
                entry = pdl_h + buffer
            elif "SHORT" not in dirs_used and prev_close >= pdl_l and row["low"] <= (pdl_l - buffer) and row["close"] <= pdl_l:
                direction = "SHORT"
                entry = pdl_l - buffer

            if direction:
                pos = open_position(sym, block["date"], direction, float(entry), float(atr), rr, float(atr), ts)
                dirs_used.add(direction)
                cd_until = ts + COOLDOWN
                activate_idx = i + 1

        eod_exit(trades, pos, float(rows.iloc[-1]["close"]))
    return trades


def simulate_pdl_retest(sym: str, blocks: list[dict], rr: float) -> list[dict]:
    trades = []
    for block in blocks:
        rows = block["today"]
        pdl_h = block["pdl_h"]
        pdl_l = block["pdl_l"]
        pos = None
        cd_until = None
        dsl = 0
        dirs_used = set()
        breakout_long = None
        breakout_short = None

        for i in range(len(rows)):
            row = rows.iloc[i]
            ts = rows.index[i]
            c = float(row["close"])
            o = float(row["open"])
            atr = row["atr"]
            if pd.isna(atr) or atr <= 0:
                continue

            if pos is not None:
                hit, pnl = check_exit(row, pos)
                if hit:
                    trades.append({**pos, "pnl": pnl, "res": hit})
                    if hit == "SL":
                        dsl += 1
                    pos = None
                continue

            prev_close = float(rows.iloc[i - 1]["close"]) if i > 0 else None
            if prev_close is None:
                continue

            if prev_close <= pdl_h and c > pdl_h:
                breakout_long = {"armed_at": i, "expires": i + 5}
            if prev_close >= pdl_l and c < pdl_l:
                breakout_short = {"armed_at": i, "expires": i + 5}

            if breakout_long and i > breakout_long["expires"]:
                breakout_long = None
            if breakout_short and i > breakout_short["expires"]:
                breakout_short = None

            if not in_window(ts) or (cd_until and ts < cd_until) or dsl >= MAX_SL_PER_DAY:
                continue

            band = float(atr) * 0.20
            direction = None
            entry = None
            stop_dist = None

            if (
                breakout_long
                and "LONG" not in dirs_used
                and i > breakout_long["armed_at"]
                and row["low"] <= (pdl_h + band)
                and c > pdl_h
                and c > o
            ):
                direction = "LONG"
                entry = c
                stop_dist = max(float(atr) * 0.9, abs(c - pdl_h) + float(atr) * 0.15)
            elif (
                breakout_short
                and "SHORT" not in dirs_used
                and i > breakout_short["armed_at"]
                and row["high"] >= (pdl_l - band)
                and c < pdl_l
                and c < o
            ):
                direction = "SHORT"
                entry = c
                stop_dist = max(float(atr) * 0.9, abs(c - pdl_l) + float(atr) * 0.15)

            if direction:
                pos = open_position(sym, block["date"], direction, float(entry), float(stop_dist), rr, float(atr), ts)
                dirs_used.add(direction)
                cd_until = ts + COOLDOWN

        eod_exit(trades, pos, float(rows.iloc[-1]["close"]))
    return trades


def simulate_orb(sym: str, blocks: list[dict], rr: float) -> list[dict]:
    trades = []
    range_end = time(9, 45)
    for block in blocks:
        rows = block["today"]
        range_rows = rows[rows.index.map(lambda x: x.time()) < range_end]
        if range_rows.empty:
            continue
        orb_high = float(range_rows["high"].max())
        orb_low = float(range_rows["low"].min())
        orb_range = orb_high - orb_low
        if orb_range <= 0:
            continue

        pos = None
        cd_until = None
        dsl = 0
        dirs_used = set()
        activate_idx = None

        for i in range(len(rows)):
            row = rows.iloc[i]
            ts = rows.index[i]
            atr = row["atr"]
            vsma = row["vol_sma"]
            if pd.isna(atr) or atr <= 0:
                continue

            if pos is not None:
                if activate_idx is not None and i >= activate_idx:
                    hit, pnl = check_exit(row, pos)
                    if hit:
                        trades.append({**pos, "pnl": pnl, "res": hit})
                        if hit == "SL":
                            dsl += 1
                        pos, activate_idx = None, None
                continue

            if ts.time() < range_end or not in_window(ts) or (cd_until and ts < cd_until) or dsl >= MAX_SL_PER_DAY:
                continue

            if pd.isna(vsma) or vsma <= 0 or row["volume"] <= vsma:
                continue

            prev_close = float(rows.iloc[i - 1]["close"]) if i > 0 else None
            if prev_close is None:
                continue

            direction = None
            entry = None
            if "LONG" not in dirs_used and prev_close <= orb_high and row["high"] >= orb_high and row["close"] >= orb_high:
                direction = "LONG"
                entry = orb_high
            elif "SHORT" not in dirs_used and prev_close >= orb_low and row["low"] <= orb_low and row["close"] <= orb_low:
                direction = "SHORT"
                entry = orb_low

            if direction:
                stop_dist = max(orb_range, float(atr) * 0.5)
                pos = open_position(sym, block["date"], direction, float(entry), stop_dist, rr, float(atr), ts)
                dirs_used.add(direction)
                cd_until = ts + COOLDOWN
                activate_idx = i + 1

        eod_exit(trades, pos, float(rows.iloc[-1]["close"]))
    return trades


def simulate_prearmed_plus_retest(sym: str, blocks: list[dict], rr: float) -> list[dict]:
    trades = []
    for block in blocks:
        rows = block["today"]
        pdl_h = block["pdl_h"]
        pdl_l = block["pdl_l"]
        pos = None
        cd_until = None
        dsl = 0
        dirs_used = set()
        breakout_long = None
        breakout_short = None
        activate_idx = None

        for i in range(len(rows)):
            row = rows.iloc[i]
            ts = rows.index[i]
            c = float(row["close"])
            o = float(row["open"])
            atr = row["atr"]
            if pd.isna(atr) or atr <= 0:
                continue

            if pos is not None:
                if activate_idx is not None and i >= activate_idx:
                    hit, pnl = check_exit(row, pos)
                    if hit:
                        trades.append({**pos, "pnl": pnl, "res": hit})
                        if hit == "SL":
                            dsl += 1
                        pos, activate_idx = None, None
                continue

            prev_close = float(rows.iloc[i - 1]["close"]) if i > 0 else None
            if prev_close is None:
                continue

            if prev_close <= pdl_h and c > pdl_h:
                breakout_long = {"armed_at": i, "expires": i + 5}
            if prev_close >= pdl_l and c < pdl_l:
                breakout_short = {"armed_at": i, "expires": i + 5}
            if breakout_long and i > breakout_long["expires"]:
                breakout_long = None
            if breakout_short and i > breakout_short["expires"]:
                breakout_short = None

            if not in_window(ts) or (cd_until and ts < cd_until) or dsl >= MAX_SL_PER_DAY:
                continue

            buffer = float(atr) * 0.10
            band = float(atr) * 0.20
            direction = None
            entry = None
            stop_dist = None

            if "LONG" not in dirs_used and prev_close <= pdl_h and row["high"] >= (pdl_h + buffer) and row["close"] >= pdl_h:
                direction = "LONG"
                entry = pdl_h + buffer
                stop_dist = float(atr)
            elif "SHORT" not in dirs_used and prev_close >= pdl_l and row["low"] <= (pdl_l - buffer) and row["close"] <= pdl_l:
                direction = "SHORT"
                entry = pdl_l - buffer
                stop_dist = float(atr)
            elif (
                breakout_long
                and "LONG" not in dirs_used
                and i > breakout_long["armed_at"]
                and row["low"] <= (pdl_h + band)
                and c > pdl_h
                and c > o
            ):
                direction = "LONG"
                entry = c
                stop_dist = max(float(atr) * 0.9, abs(c - pdl_h) + float(atr) * 0.15)
            elif (
                breakout_short
                and "SHORT" not in dirs_used
                and i > breakout_short["armed_at"]
                and row["high"] >= (pdl_l - band)
                and c < pdl_l
                and c < o
            ):
                direction = "SHORT"
                entry = c
                stop_dist = max(float(atr) * 0.9, abs(c - pdl_l) + float(atr) * 0.15)

            if direction:
                pos = open_position(sym, block["date"], direction, float(entry), float(stop_dist), rr, float(atr), ts)
                dirs_used.add(direction)
                cd_until = ts + COOLDOWN
                activate_idx = i + 1

        eod_exit(trades, pos, float(rows.iloc[-1]["close"]))
    return trades


def simulate_orb_plus_retest(sym: str, blocks: list[dict], rr: float) -> list[dict]:
    trades = []
    range_end = time(9, 45)
    for block in blocks:
        rows = block["today"]
        pdl_h = block["pdl_h"]
        pdl_l = block["pdl_l"]
        range_rows = rows[rows.index.map(lambda x: x.time()) < range_end]
        if range_rows.empty:
            continue
        orb_high = float(range_rows["high"].max())
        orb_low = float(range_rows["low"].min())
        orb_range = orb_high - orb_low
        if orb_range <= 0:
            continue

        pos = None
        cd_until = None
        dsl = 0
        dirs_used = set()
        breakout_long = None
        breakout_short = None
        activate_idx = None

        for i in range(len(rows)):
            row = rows.iloc[i]
            ts = rows.index[i]
            c = float(row["close"])
            o = float(row["open"])
            atr = row["atr"]
            vsma = row["vol_sma"]
            if pd.isna(atr) or atr <= 0:
                continue

            if pos is not None:
                if activate_idx is not None and i >= activate_idx:
                    hit, pnl = check_exit(row, pos)
                    if hit:
                        trades.append({**pos, "pnl": pnl, "res": hit})
                        if hit == "SL":
                            dsl += 1
                        pos, activate_idx = None, None
                continue

            prev_close = float(rows.iloc[i - 1]["close"]) if i > 0 else None
            if prev_close is None:
                continue

            if prev_close <= pdl_h and c > pdl_h:
                breakout_long = {"armed_at": i, "expires": i + 5}
            if prev_close >= pdl_l and c < pdl_l:
                breakout_short = {"armed_at": i, "expires": i + 5}
            if breakout_long and i > breakout_long["expires"]:
                breakout_long = None
            if breakout_short and i > breakout_short["expires"]:
                breakout_short = None

            if not in_window(ts) or (cd_until and ts < cd_until) or dsl >= MAX_SL_PER_DAY:
                continue

            band = float(atr) * 0.20
            direction = None
            entry = None
            stop_dist = None

            if ts.time() >= range_end and not pd.isna(vsma) and vsma > 0 and row["volume"] > vsma:
                if "LONG" not in dirs_used and prev_close <= orb_high and row["high"] >= orb_high and row["close"] >= orb_high:
                    direction = "LONG"
                    entry = orb_high
                    stop_dist = max(orb_range, float(atr) * 0.5)
                elif "SHORT" not in dirs_used and prev_close >= orb_low and row["low"] <= orb_low and row["close"] <= orb_low:
                    direction = "SHORT"
                    entry = orb_low
                    stop_dist = max(orb_range, float(atr) * 0.5)

            if direction is None:
                if (
                    breakout_long
                    and "LONG" not in dirs_used
                    and i > breakout_long["armed_at"]
                    and row["low"] <= (pdl_h + band)
                    and c > pdl_h
                    and c > o
                ):
                    direction = "LONG"
                    entry = c
                    stop_dist = max(float(atr) * 0.9, abs(c - pdl_h) + float(atr) * 0.15)
                elif (
                    breakout_short
                    and "SHORT" not in dirs_used
                    and i > breakout_short["armed_at"]
                    and row["high"] >= (pdl_l - band)
                    and c < pdl_l
                    and c < o
                ):
                    direction = "SHORT"
                    entry = c
                    stop_dist = max(float(atr) * 0.9, abs(c - pdl_l) + float(atr) * 0.15)

            if direction:
                pos = open_position(sym, block["date"], direction, float(entry), float(stop_dist), rr, float(atr), ts)
                dirs_used.add(direction)
                cd_until = ts + COOLDOWN
                activate_idx = i + 1

        eod_exit(trades, pos, float(rows.iloc[-1]["close"]))
    return trades


STRATEGIES = {
    "Current PDL": simulate_current_pdl,
    "Prearmed PDL": simulate_prearmed_pdl,
    "PDL Retest": simulate_pdl_retest,
    "ORB": simulate_orb,
    "Prearmed + Retest": simulate_prearmed_plus_retest,
    "ORB + Retest": simulate_orb_plus_retest,
}


def run():
    print("=" * 140)
    print("SCALPING STRATEGY SHOOTOUT — 90 days, 48 Nifty stocks, 3-min candles, no friction")
    print("=" * 140)
    print(f"Entry window: {ENTRY_START:%H:%M}-{ENTRY_END:%H:%M}")
    print(f"RR sweep: {RRS}")
    print()

    stock_blocks = {}
    sectors = {}
    for sym, sec in NIFTY50:
        p = csv_1min(sym)
        if not p.exists():
            continue
        try:
            df = load_csv(str(p))
            df3 = resample_3min(df)
            ind = compute_indicators(df3)
            blocks = get_day_blocks(ind)
            if blocks:
                stock_blocks[sym] = blocks
                sectors[sym] = sec
        except Exception:
            continue

    print(f"Loaded {len(stock_blocks)} stocks with 3-min derived history.\n")

    best_rows = []
    for name, fn in STRATEGIES.items():
        best = None
        print(f"[{name}]")
        for rr in RRS:
            stock_trades = {}
            for sym, blocks in stock_blocks.items():
                stock_trades[sym] = fn(sym, blocks, rr)
            row = summarize_strategy(name, rr, stock_trades, sectors)
            print(
                f"  RR={rr:>4.2f}  all48=₹{row['overall']['pnl']:>9,.0f} "
                f"top10=₹{row['top10']['pnl']:>8,.0f} recent10=₹{row['recent10']['pnl']:>7,.0f} "
                f"WR={row['overall']['wr']:>5.1f}% DD=₹{row['overall']['dd']:>7,.0f}"
            )
            if best is None or row["top10"]["pnl"] > best["top10"]["pnl"]:
                best = row
        best_rows.append(best)
        print()

    print("=" * 140)
    print("BEST VERSION OF EACH STRATEGY — ranked by 90d TOP-10 P&L")
    print("=" * 140)
    best_rows.sort(key=lambda r: r["top10"]["pnl"], reverse=True)
    print(
        f"  {'#':>2} {'Strategy':<20} {'RR':>4} │ {'All48 P&L':>10} {'Top10 P&L':>10} {'Recent10':>9} │ "
        f"{'Trades':>6} {'WR%':>6} {'DD':>9} {'PF':>5} {'Avg':>7}"
    )
    print("  " + "─" * 120)
    for i, r in enumerate(best_rows, 1):
        o = r["overall"]
        t = r["top10"]
        recent = r["recent10"]
        print(
            f"  {i:>2} {r['strategy']:<20} {r['rr']:>4.2f} │ "
            f"₹{o['pnl']:>8,.0f} ₹{t['pnl']:>8,.0f} ₹{recent['pnl']:>7,.0f} │ "
            f"{o['n']:>6} {o['wr']:>5.1f}% ₹{o['dd']:>7,.0f} {o['pf']:>5.2f} ₹{o['avg']:>5,.0f}"
        )

    print("\n" + "=" * 140)
    print("BEST VERSION OF EACH STRATEGY — ranked by RECENT 10-DAY TOP-10 P&L")
    print("=" * 140)
    recent_rank = sorted(best_rows, key=lambda r: r["recent10"]["pnl"], reverse=True)
    for i, r in enumerate(recent_rank, 1):
        print(
            f"  {i:>2}. {r['strategy']:<20} RR={r['rr']:.2f}  "
            f"recent10=₹{r['recent10']['pnl']:>7,.0f}  "
            f"90d top10=₹{r['top10']['pnl']:>8,.0f}  all48=₹{r['overall']['pnl']:>9,.0f}"
        )

    winner = best_rows[0]
    print("\n" + "=" * 140)
    print(f"WINNER — {winner['strategy']} (RR={winner['rr']:.2f})")
    print("=" * 140)
    print(f"All 48 P&L: ₹{winner['overall']['pnl']:,.0f}")
    print(f"Top 10 P&L: ₹{winner['top10']['pnl']:,.0f}")
    print(f"Recent 10-day top10 P&L: ₹{winner['recent10']['pnl']:,.0f}")
    print(f"Trades: {winner['overall']['n']}  WR: {winner['overall']['wr']:.1f}%  DD: ₹{winner['overall']['dd']:,.0f}")

    ranked = winner["ranked"]
    print("\nTop 10 stocks for winner:")
    print(f"  {'#':>2} {'Stock':<14} {'Sector':<12} {'P&L':>10}")
    print("  " + "─" * 45)
    for i, (sym, pnl) in enumerate(ranked[:10], 1):
        print(f"  {i:>2} {sym:<14} {winner['sectors'][sym]:<12} ₹{pnl:>8,.0f}")

    print("\nBottom 5:")
    for sym, pnl in ranked[-5:]:
        print(f"  {sym:<14} {winner['sectors'][sym]:<12} ₹{pnl:>8,.0f}")

    print("\nStock config for winner:")
    print("  STOCKS = [")
    for sym, _ in ranked[:10]:
        print(f'      ("{sym}", "{winner["sectors"][sym]}"),')
    print("  ]")
    print("=" * 140)


if __name__ == "__main__":
    run()

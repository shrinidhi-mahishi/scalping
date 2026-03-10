"""
Reality-check backtest for live-vs-backtest gap.

What this tests:
1. Optimistic execution: enter at signal bar close
2. Delayed execution: enter at next bar open
3. Realistic execution: next bar open + slippage + fees + EOD square-off

Strategies:
  - Current PDL strategy from live_signals.py
  - Best ORB variant discovered so far:
      3-min candle, 30-min range, RR=0.75, volume filter ON, entry until 13:00

Universes:
  - Current live PDL top 10 stocks
  - Same current top 10 stocks but traded with ORB logic
  - ORB-optimized top 10 stocks
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from fetch_data import load_csv

CASH = 50_000.0
TODAY = pd.Timestamp.now().normalize()
START_90D = TODAY - pd.Timedelta(days=90)
FEE_BPS_REALISTIC = 2.0
SLIP_BPS_REALISTIC = 5.0
_F = str.maketrans({"&": ""})


PDL_STOCKS = [
    ("TATAMOTORS", "Auto"),
    ("TATASTEEL", "Steel"),
    ("HINDALCO", "Metals"),
    ("HINDUNILVR", "FMCG"),
    ("INFY", "IT"),
    ("HDFCBANK", "Banking"),
    ("ADANIENT", "Infra"),
    ("JSWSTEEL", "Steel"),
    ("AXISBANK", "Banking"),
    ("SHRIRAMFIN", "Finance"),
]

ORB_STOCKS = [
    ("TATACONSUM", "Consumer"),
    ("ITC", "FMCG"),
    ("BAJAJ-AUTO", "Auto"),
    ("TCS", "IT"),
    ("AXISBANK", "Banking"),
    ("ADANIPORTS", "Ports"),
    ("SBILIFE", "Insurance"),
    ("TITAN", "Consumer"),
    ("BAJAJFINSV", "Finance"),
    ("EICHERMOT", "Auto"),
]


@dataclass(frozen=True)
class Scenario:
    name: str
    next_bar_entry: bool
    slip_bps: float
    fee_bps: float
    force_eod_exit: bool


SCENARIOS = [
    Scenario(
        name="same_bar_close",
        next_bar_entry=False,
        slip_bps=0.0,
        fee_bps=0.0,
        force_eod_exit=True,
    ),
    Scenario(
        name="next_bar_open",
        next_bar_entry=True,
        slip_bps=0.0,
        fee_bps=0.0,
        force_eod_exit=True,
    ),
    Scenario(
        name="realistic",
        next_bar_entry=True,
        slip_bps=SLIP_BPS_REALISTIC,
        fee_bps=FEE_BPS_REALISTIC,
        force_eod_exit=True,
    ),
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


def compute_pdl_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    tp = (out["high"] + out["low"] + out["close"]) / 3.0
    tp_vol = tp * out["volume"]
    day = out.index.date
    out["vwap"] = tp_vol.groupby(day).cumsum() / out["volume"].groupby(day).cumsum()

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


def compute_orb_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pc = out["close"].shift(1)
    tr = np.maximum(out["high"], pc) - np.minimum(out["low"], pc)
    out["atr"] = tr.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
    out["vol_sma"] = out["volume"].rolling(20).mean()
    return out


def get_day_blocks(ind: pd.DataFrame) -> list[dict]:
    days = sorted(ind.index.normalize().unique())
    blocks: list[dict] = []
    for i in range(1, len(days)):
        td = days[i]
        pd_ = days[i - 1]
        prev_rows = ind[ind.index.normalize() == pd_]
        today_rows = ind[ind.index.normalize() == td]
        if prev_rows.empty or today_rows.empty:
            continue
        blocks.append(
            {
                "date": td,
                "prev": prev_rows,
                "today": today_rows,
                "pdl_h": float(prev_rows["high"].max()),
                "pdl_l": float(prev_rows["low"].min()),
            }
        )
    return blocks


def buy_fill(price: float, slip_bps: float) -> float:
    return price * (1.0 + slip_bps / 10_000.0)


def sell_fill(price: float, slip_bps: float) -> float:
    return price * (1.0 - slip_bps / 10_000.0)


def adverse_entry_fill(direction: str, raw_price: float, slip_bps: float) -> float:
    return buy_fill(raw_price, slip_bps) if direction == "LONG" else sell_fill(raw_price, slip_bps)


def adverse_exit_fill(direction: str, raw_price: float, slip_bps: float) -> float:
    return sell_fill(raw_price, slip_bps) if direction == "LONG" else buy_fill(raw_price, slip_bps)


def pnl_after_friction(direction: str, entry_fill: float, exit_fill: float, qty: int, fee_bps: float) -> float:
    gross = (exit_fill - entry_fill) * qty if direction == "LONG" else (entry_fill - exit_fill) * qty
    fee_rate = fee_bps / 10_000.0
    fees = (entry_fill + exit_fill) * qty * fee_rate
    return gross - fees


def summarize(trades: list[dict], meta: dict) -> dict:
    if not trades:
        return {
            "n": 0,
            "wr": 0.0,
            "pnl": 0.0,
            "tp": 0,
            "sl": 0,
            "eod": 0,
            "dd": 0.0,
            "pf": 0.0,
            "avg": 0.0,
            "delay_cash": 0.0,
            "delay_per_trade": 0.0,
            **meta,
        }
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
    delay_cash = float(tdf.get("delay_cash", pd.Series(dtype=float)).sum())
    return {
        "n": n,
        "wr": wins / n * 100.0,
        "pnl": pnl,
        "tp": tp,
        "sl": sl,
        "eod": eod,
        "dd": dd,
        "pf": min(pf, 9.99),
        "avg": pnl / n,
        "delay_cash": delay_cash,
        "delay_per_trade": delay_cash / n,
        **meta,
    }


def format_summary(r: dict) -> str:
    return (
        f"{r['scenario']:<15}  trades={r['n']:>4}  WR={r['wr']:>5.1f}%  "
        f"P&L=₹{r['pnl']:>8,.0f}  DD=₹{r['dd']:>7,.0f}  PF={r['pf']:.2f}  "
        f"Avg=₹{r['avg']:>5,.0f}  EOD={r['eod']:>3}  DelayCost=₹{r['delay_cash']:>7,.0f}"
    )


def simulate_pdl(blocks: list[dict], symbol: str, scenario: Scenario) -> tuple[list[dict], dict]:
    RR = 2.5
    RISK_PCT = 0.015
    LEV_CAP = 5.0
    PDL_SL_MULT = 1.2
    ENTRY_START = time(9, 30)
    ENTRY_END = time(14, 30)
    COOLDOWN = timedelta(minutes=30)
    MAX_SL_PER_DAY = 2

    trades: list[dict] = []
    skipped_gap = 0

    for block in blocks:
        rows = block["today"]
        pdl_h = block["pdl_h"]
        pdl_l = block["pdl_l"]
        pos = None
        cd_until = None
        dsl = 0
        pdl_dirs: set[str] = set()

        for i in range(len(rows)):
            row = rows.iloc[i]
            ts = rows.index[i]
            t = ts.time()
            c = float(row["close"])
            o = float(row["open"])

            if pos is not None and i >= pos["active_from"]:
                hit = None
                exit_base = None
                if pos["d"] == "LONG":
                    if row["low"] <= pos["sl"]:
                        hit = "SL"
                        exit_base = pos["sl"]
                    elif row["high"] >= pos["tp"]:
                        hit = "TP"
                        exit_base = pos["tp"]
                else:
                    if row["high"] >= pos["sl"]:
                        hit = "SL"
                        exit_base = pos["sl"]
                    elif row["low"] <= pos["tp"]:
                        hit = "TP"
                        exit_base = pos["tp"]

                if hit is not None:
                    exit_fill = adverse_exit_fill(pos["d"], exit_base, scenario.slip_bps)
                    pnl = pnl_after_friction(pos["d"], pos["entry_fill"], exit_fill, pos["q"], scenario.fee_bps)
                    trades.append(
                        {
                            **pos,
                            "exit_fill": exit_fill,
                            "pnl": pnl,
                            "res": hit,
                        }
                    )
                    if hit == "SL":
                        dsl += 1
                    pos = None
                    continue

            prev_close = float(rows.iloc[i - 1]["close"]) if i > 0 else None
            if pos is not None:
                continue
            if not (ENTRY_START <= t <= ENTRY_END):
                continue
            if cd_until and ts < cd_until:
                continue
            if dsl >= MAX_SL_PER_DAY:
                continue

            rsi = row["rsi"]
            atr = row["atr"]
            vwap = row["vwap"]
            vsma = row["vol_sma"]
            if pd.isna(rsi) or pd.isna(atr) or atr <= 0 or pd.isna(vwap) or pd.isna(vsma):
                continue

            vol_ok = bool(row["volume"] > vsma) if vsma > 0 else False
            direction = None
            if prev_close is not None and vol_ok:
                if (
                    "LONG" not in pdl_dirs
                    and prev_close <= pdl_h
                    and c > pdl_h
                    and c > vwap
                    and rsi > 50
                ):
                    direction = "LONG"
                elif (
                    "SHORT" not in pdl_dirs
                    and prev_close >= pdl_l
                    and c < pdl_l
                    and c < vwap
                    and rsi < 50
                ):
                    direction = "SHORT"

            if direction is None:
                continue

            signal_price = c
            stop_dist = float(atr) * PDL_SL_MULT
            sl = signal_price - stop_dist if direction == "LONG" else signal_price + stop_dist
            tp = signal_price + stop_dist * RR if direction == "LONG" else signal_price - stop_dist * RR
            risk_qty = int(CASH * RISK_PCT / stop_dist)
            max_qty = int(CASH * LEV_CAP / signal_price)
            qty = max(min(risk_qty, max_qty), 1)

            if scenario.next_bar_entry:
                if i + 1 >= len(rows):
                    continue
                next_open = float(rows.iloc[i + 1]["open"])
                if direction == "LONG" and (next_open >= tp or next_open <= sl):
                    skipped_gap += 1
                    continue
                if direction == "SHORT" and (next_open <= tp or next_open >= sl):
                    skipped_gap += 1
                    continue
                entry_fill = adverse_entry_fill(direction, next_open, scenario.slip_bps)
                delay_ps = (entry_fill - signal_price) if direction == "LONG" else (signal_price - entry_fill)
                active_from = i + 1
            else:
                entry_fill = adverse_entry_fill(direction, signal_price, scenario.slip_bps)
                delay_ps = 0.0
                active_from = i + 1

            pos = {
                "sym": symbol,
                "d": direction,
                "signal_price": signal_price,
                "entry_fill": entry_fill,
                "sl": sl,
                "tp": tp,
                "atr": float(atr),
                "q": qty,
                "date": block["date"],
                "active_from": active_from,
                "delay_cash": max(delay_ps, 0.0) * qty,
                "delay_ps": delay_ps,
            }
            cd_until = ts + COOLDOWN
            pdl_dirs.add(direction)

        if pos is not None and scenario.force_eod_exit:
            last_close = float(rows.iloc[-1]["close"])
            exit_fill = adverse_exit_fill(pos["d"], last_close, scenario.slip_bps)
            pnl = pnl_after_friction(pos["d"], pos["entry_fill"], exit_fill, pos["q"], scenario.fee_bps)
            trades.append(
                {
                    **pos,
                    "exit_fill": exit_fill,
                    "pnl": pnl,
                    "res": "EOD",
                }
            )

    return trades, {"skipped_gap": skipped_gap}


def simulate_orb(blocks: list[dict], symbol: str, scenario: Scenario) -> tuple[list[dict], dict]:
    RR = 0.75
    RISK_PCT = 0.015
    LEV_CAP = 5.0
    ENTRY_END = time(13, 0)
    RANGE_END = time(9, 45)
    COOLDOWN = timedelta(minutes=30)
    MAX_SL_PER_DAY = 2

    trades: list[dict] = []
    skipped_gap = 0

    for block in blocks:
        rows = block["today"]
        range_rows = rows[rows.index.map(lambda x: x.time()) < RANGE_END]
        if range_rows.empty:
            continue
        orb_high = float(range_rows["high"].max())
        orb_low = float(range_rows["low"].min())
        orb_range = orb_high - orb_low
        if orb_range <= 0:
            continue

        pos = None
        dsl = 0
        traded_dirs: set[str] = set()
        cd_until = None

        for i in range(len(rows)):
            row = rows.iloc[i]
            ts = rows.index[i]
            t = ts.time()
            c = float(row["close"])

            if pos is not None and i >= pos["active_from"]:
                hit = None
                exit_base = None
                if pos["d"] == "LONG":
                    if row["low"] <= pos["sl"]:
                        hit = "SL"
                        exit_base = pos["sl"]
                    elif row["high"] >= pos["tp"]:
                        hit = "TP"
                        exit_base = pos["tp"]
                else:
                    if row["high"] >= pos["sl"]:
                        hit = "SL"
                        exit_base = pos["sl"]
                    elif row["low"] <= pos["tp"]:
                        hit = "TP"
                        exit_base = pos["tp"]
                if hit is not None:
                    exit_fill = adverse_exit_fill(pos["d"], exit_base, scenario.slip_bps)
                    pnl = pnl_after_friction(pos["d"], pos["entry_fill"], exit_fill, pos["q"], scenario.fee_bps)
                    trades.append(
                        {
                            **pos,
                            "exit_fill": exit_fill,
                            "pnl": pnl,
                            "res": hit,
                        }
                    )
                    if hit == "SL":
                        dsl += 1
                    pos = None
                    continue

            if pos is not None:
                continue
            if t < RANGE_END or t > ENTRY_END:
                continue
            if dsl >= MAX_SL_PER_DAY:
                continue
            if cd_until and ts < cd_until:
                continue

            atr = row["atr"]
            vsma = row["vol_sma"]
            if pd.isna(atr) or atr <= 0 or pd.isna(vsma):
                continue
            if vsma <= 0 or row["volume"] <= vsma:
                continue

            direction = None
            if "LONG" not in traded_dirs and c > orb_high:
                direction = "LONG"
            elif "SHORT" not in traded_dirs and c < orb_low:
                direction = "SHORT"
            if direction is None:
                continue

            signal_price = c
            stop_dist = max(orb_range, float(atr) * 0.5)
            sl = signal_price - stop_dist if direction == "LONG" else signal_price + stop_dist
            tp = signal_price + stop_dist * RR if direction == "LONG" else signal_price - stop_dist * RR
            risk_qty = int(CASH * RISK_PCT / stop_dist)
            max_qty = int(CASH * LEV_CAP / signal_price)
            qty = max(min(risk_qty, max_qty), 1)

            if scenario.next_bar_entry:
                if i + 1 >= len(rows):
                    continue
                next_open = float(rows.iloc[i + 1]["open"])
                if direction == "LONG" and (next_open >= tp or next_open <= sl):
                    skipped_gap += 1
                    continue
                if direction == "SHORT" and (next_open <= tp or next_open >= sl):
                    skipped_gap += 1
                    continue
                entry_fill = adverse_entry_fill(direction, next_open, scenario.slip_bps)
                delay_ps = (entry_fill - signal_price) if direction == "LONG" else (signal_price - entry_fill)
                active_from = i + 1
            else:
                entry_fill = adverse_entry_fill(direction, signal_price, scenario.slip_bps)
                delay_ps = 0.0
                active_from = i + 1

            pos = {
                "sym": symbol,
                "d": direction,
                "signal_price": signal_price,
                "entry_fill": entry_fill,
                "sl": sl,
                "tp": tp,
                "atr": float(atr),
                "q": qty,
                "date": block["date"],
                "active_from": active_from,
                "delay_cash": max(delay_ps, 0.0) * qty,
                "delay_ps": delay_ps,
            }
            traded_dirs.add(direction)
            cd_until = ts + COOLDOWN

        if pos is not None and scenario.force_eod_exit:
            last_close = float(rows.iloc[-1]["close"])
            exit_fill = adverse_exit_fill(pos["d"], last_close, scenario.slip_bps)
            pnl = pnl_after_friction(pos["d"], pos["entry_fill"], exit_fill, pos["q"], scenario.fee_bps)
            trades.append(
                {
                    **pos,
                    "exit_fill": exit_fill,
                    "pnl": pnl,
                    "res": "EOD",
                }
            )

    return trades, {"skipped_gap": skipped_gap}


def load_blocks(stocks: list[tuple[str, str]], indicator_kind: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for sym, _ in stocks:
        path = csv_1min(sym)
        if not path.exists():
            continue
        df = load_csv(str(path))
        df = df[df.index >= START_90D - pd.Timedelta(days=10)]
        df3 = resample_3min(df)
        if indicator_kind == "pdl":
            ind = compute_pdl_indicators(df3)
        else:
            ind = compute_orb_indicators(df3)
        blocks = get_day_blocks(ind)
        if blocks:
            out[sym] = blocks
    return out


def run_suite(
    title: str,
    stocks: list[tuple[str, str]],
    simulator,
    indicator_kind: str,
    recent_days: int | None = None,
) -> list[dict]:
    stock_blocks = load_blocks(stocks, indicator_kind)
    results = []

    if recent_days is not None:
        for sym in list(stock_blocks):
            stock_blocks[sym] = stock_blocks[sym][-recent_days:]

    print("=" * 120)
    print(title)
    print("=" * 120)
    print(f"  Stocks loaded: {len(stock_blocks)}")
    if recent_days is not None:
        print(f"  Window: last {recent_days} trading days")
    print()

    for scenario in SCENARIOS:
        all_trades = []
        skipped_gap = 0
        for sym in stock_blocks:
            trades, meta = simulator(stock_blocks[sym], sym, scenario)
            all_trades.extend(trades)
            skipped_gap += meta["skipped_gap"]
        summary = summarize(
            all_trades,
            {
                "scenario": scenario.name,
                "skipped_gap": skipped_gap,
            },
        )
        results.append(summary)
        print(format_summary(summary) + f"  SkippedGap={skipped_gap:>3}")
    print()
    return results


def print_delta(title: str, results: list[dict]) -> None:
    by_name = {r["scenario"]: r for r in results}
    base = by_name["same_bar_close"]
    delayed = by_name["next_bar_open"]
    real = by_name["realistic"]
    print(title)
    print(
        f"  same_bar -> next_bar:  P&L {delayed['pnl'] - base['pnl']:+,.0f}, "
        f"WR {delayed['wr'] - base['wr']:+.1f}%, DD {delayed['dd'] - base['dd']:+,.0f}"
    )
    print(
        f"  next_bar -> realistic: P&L {real['pnl'] - delayed['pnl']:+,.0f}, "
        f"WR {real['wr'] - delayed['wr']:+.1f}%, DD {real['dd'] - delayed['dd']:+,.0f}"
    )
    print(
        f"  same_bar -> realistic: P&L {real['pnl'] - base['pnl']:+,.0f}, "
        f"DelayCost=₹{real['delay_cash']:,.0f}, SkippedGap={real['skipped_gap']}"
    )
    print()


def run() -> None:
    print("=" * 120)
    print("REALITY CHECK — PDL vs ORB under delayed execution")
    print("=" * 120)
    print(f"Assumptions:")
    print(f"  realistic slippage: {SLIP_BPS_REALISTIC:.1f} bps/side")
    print(f"  realistic fees:     {FEE_BPS_REALISTIC:.1f} bps/side")
    print(f"  square-off:         end of day at final close")
    print()

    pdl_90 = run_suite(
        "PDL on current live stock list — 90 days",
        PDL_STOCKS,
        simulate_pdl,
        indicator_kind="pdl",
    )
    print_delta("PDL degradation summary — 90 days", pdl_90)

    pdl_10 = run_suite(
        "PDL on current live stock list — recent 10 trading days",
        PDL_STOCKS,
        simulate_pdl,
        indicator_kind="pdl",
        recent_days=10,
    )
    print_delta("PDL degradation summary — recent 10 days", pdl_10)

    orb_same_universe_90 = run_suite(
        "ORB on CURRENT PDL stock list — 90 days",
        PDL_STOCKS,
        simulate_orb,
        indicator_kind="orb",
    )
    print_delta("ORB degradation summary on current stock list — 90 days", orb_same_universe_90)

    orb_best_universe_90 = run_suite(
        "ORB on ORB-optimized stock list — 90 days",
        ORB_STOCKS,
        simulate_orb,
        indicator_kind="orb",
    )
    print_delta("ORB degradation summary on ORB stock list — 90 days", orb_best_universe_90)

    print("=" * 120)
    print("Bottom line")
    print("=" * 120)
    print("If PDL loses most of its edge when entry shifts from same-bar close to next-bar open,")
    print("then the strategy is structurally too late for live trading. If ORB degrades less,")
    print("that is strong evidence the problem is strategy timing, not infrastructure speed.")
    print("=" * 120)


if __name__ == "__main__":
    run()

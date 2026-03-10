"""
Multi-Strategy 1-Min Backtest on Nifty 50 — 90 days.

Strategies tested:
  1. PDL-Scaled  — Prev-Day Level breakout with indicator periods scaled for 1-min
  2. ORB         — Opening Range Breakout (first 15 min high/low)
  3. VWAP Bounce — Mean-reversion entries on VWAP pullbacks with volume
  4. EMA Scalp   — EMA-9/21 crossover momentum scalping

Each strategy is tested with RR sweep (1.0–3.0).
Top 10 stocks selected per strategy.
"""

import sys
import time as _time
from datetime import time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(".").resolve()))
from fetch_data import AngelOneClient, load_csv, save_csv

# ─── Config ──────────────────────────────────────────────────────────────

CASH = 50_000.0
TODAY = pd.Timestamp.now().normalize()
START = TODAY - pd.Timedelta(days=90)
_F = str.maketrans({"&": ""})

RISK_PCT = 0.015
LEV_CAP = 5.0
MAX_SL_PER_DAY = 2
COOLDOWN_MIN = 30
ENTRY_START = time(9, 30)
ENTRY_END = time(13, 0)

RR_VALUES = [1.0, 1.5, 2.0, 2.5, 3.0]

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


# ─── Data Helpers ────────────────────────────────────────────────────────

def csv_1min(sym):
    return Path("data") / f"{sym.translate(_F)}_1min.csv"


def fetch_all_1min():
    needs = []
    for sym, _ in NIFTY50:
        p = csv_1min(sym)
        if p.exists():
            df = load_csv(str(p))
            if df.index.max().date() >= (TODAY - timedelta(days=2)).date():
                print(f"  {sym:14s} — cached ({len(df):,} bars)")
                continue
        needs.append(sym)

    if not needs:
        print("  All 48 stocks cached.\n")
        return

    print(f"\n  {len(needs)} stocks need fetching...\n")
    client = AngelOneClient()
    client.connect()

    for i, sym in enumerate(needs):
        print(f"  [{i+1}/{len(needs)}] {sym:14s} ", end="", flush=True)
        try:
            df = client.fetch_history(sym, days=100, interval="ONE_MINUTE")
            Path("data").mkdir(exist_ok=True)
            save_csv(df, str(csv_1min(sym)))
        except Exception as e:
            print(f"FAILED: {e}")
        _time.sleep(1)
    print()


# ─── Indicator Engine ────────────────────────────────────────────────────

def compute_indicators(df, rsi_p=9, atr_p=14, vol_p=20):
    out = df.copy()
    tp = (out["high"] + out["low"] + out["close"]) / 3.0
    day = out.index.date
    out["vwap"] = (tp * out["volume"]).groupby(day).cumsum() / out["volume"].groupby(day).cumsum()

    delta = out["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_g = gain.ewm(alpha=1.0 / rsi_p, min_periods=rsi_p, adjust=False).mean()
    avg_l = loss.ewm(alpha=1.0 / rsi_p, min_periods=rsi_p, adjust=False).mean()
    rs = avg_g / avg_l.replace(0, np.nan)
    out["rsi"] = 100.0 - 100.0 / (1.0 + rs)

    out["vol_sma"] = out["volume"].rolling(vol_p).mean()
    pc = out["close"].shift(1)
    tr = np.maximum(out["high"], pc) - np.minimum(out["low"], pc)
    out["atr"] = tr.ewm(alpha=1.0 / atr_p, min_periods=atr_p, adjust=False).mean()

    out["ema9"] = out["close"].ewm(span=9, adjust=False).mean()
    out["ema21"] = out["close"].ewm(span=21, adjust=False).mean()

    return out


def get_day_groups(ind):
    """Return list of (trade_date, prev_day_data, today_data, pdl_h, pdl_l)."""
    days = sorted(ind.index.normalize().unique())
    out = []
    for i in range(1, len(days)):
        td, pd_ = days[i], days[i - 1]
        pr = ind[ind.index.normalize() == pd_]
        tr = ind[ind.index.normalize() == td]
        if pr.empty or tr.empty:
            continue
        pdl_h = float(pr["high"].max())
        pdl_l = float(pr["low"].min())
        out.append((td, pr, tr, pdl_h, pdl_l))
    return out


# ─── Trade Execution Engine ──────────────────────────────────────────────

def execute_trades(signals, rows, rr, sl_mult, max_sl=MAX_SL_PER_DAY):
    """Given a list of signal dicts, walk through bars and resolve SL/TP.
    Each signal: {idx, direction, entry_price, atr, sl_mult}
    """
    trades = []
    pos = None
    dsl = 0

    for i in range(len(rows)):
        row = rows.iloc[i]
        ts = rows.index[i]

        if pos is not None:
            hit, pv = None, 0
            if pos["d"] == "LONG":
                if row["low"] <= pos["sl"]:
                    hit, pv = "SL", (pos["sl"] - pos["e"]) * pos["q"]
                elif row["high"] >= pos["tp"]:
                    hit, pv = "TP", (pos["tp"] - pos["e"]) * pos["q"]
            else:
                if row["high"] >= pos["sl"]:
                    hit, pv = "SL", (pos["e"] - pos["sl"]) * pos["q"]
                elif row["low"] <= pos["tp"]:
                    hit, pv = "TP", (pos["e"] - pos["tp"]) * pos["q"]
            if hit:
                trades.append({**pos, "pnl": pv, "res": hit})
                if hit == "SL":
                    dsl += 1
                pos = None
            continue

        if dsl >= max_sl:
            continue

        for sig in signals:
            if sig["idx"] == i:
                c = sig["entry"]
                atr = sig["atr"]
                sd = atr * sig["sl_mult"]
                rq = int(CASH * RISK_PCT / sd)
                mq = int(CASH * LEV_CAP / c)
                q = max(min(rq, mq), 1)
                if sig["d"] == "LONG":
                    slp, tpp = c - sd, c + sd * rr
                else:
                    slp, tpp = c + sd, c - sd * rr
                pos = {"sym": sig["sym"], "d": sig["d"], "e": c,
                       "sl": slp, "tp": tpp, "atr": atr, "q": q,
                       "date": sig["date"], "h": ts.time().hour}
                break
    return trades


# ─── Strategy 1: PDL-Scaled ──────────────────────────────────────────────

def strat_pdl_scaled(sym, day_groups, rr):
    """PDL breakout with 1-min-scaled indicators (RSI-27, ATR-42, VOL-60)."""
    SL_MULT = 1.2
    trades_all = []
    for td, _, today, pdl_h, pdl_l in day_groups:
        pdl_dirs = set()
        cd_until = None
        pos, pc, dsl = None, None, 0
        cooldown = timedelta(minutes=COOLDOWN_MIN)

        for i in range(len(today)):
            row = today.iloc[i]
            ts = today.index[i]
            t = ts.time()
            c = row["close"]

            if pos is not None:
                hit, pv = None, 0
                if pos["d"] == "LONG":
                    if row["low"] <= pos["sl"]:
                        hit, pv = "SL", (pos["sl"] - pos["e"]) * pos["q"]
                    elif row["high"] >= pos["tp"]:
                        hit, pv = "TP", (pos["tp"] - pos["e"]) * pos["q"]
                else:
                    if row["high"] >= pos["sl"]:
                        hit, pv = "SL", (pos["e"] - pos["sl"]) * pos["q"]
                    elif row["low"] <= pos["tp"]:
                        hit, pv = "TP", (pos["e"] - pos["tp"]) * pos["q"]
                if hit:
                    trades_all.append({**pos, "pnl": pv, "res": hit})
                    if hit == "SL":
                        dsl += 1
                    pos, pc = None, c
                    continue
                pc = c
                continue

            if not (ENTRY_START <= t <= ENTRY_END):
                pc = c
                continue
            if cd_until and ts < cd_until:
                pc = c
                continue
            if dsl >= MAX_SL_PER_DAY:
                pc = c
                continue

            rsi = row["rsi"]
            atr = row["atr"]
            vwap = row["vwap"]
            vsma = row["vol_sma"]
            if pd.isna(rsi) or pd.isna(atr) or atr <= 0 or pd.isna(vwap) or pd.isna(vsma):
                pc = c
                continue

            vol_ok = row["volume"] > vsma if vsma > 0 else False
            sig = None

            if pc is not None and vol_ok:
                if ("LONG" not in pdl_dirs and pc <= pdl_h and c > pdl_h
                        and c > vwap and rsi > 50):
                    sig = "LONG"
                elif ("SHORT" not in pdl_dirs and pc >= pdl_l and c < pdl_l
                      and c < vwap and rsi < 50):
                    sig = "SHORT"

            if sig:
                sd = atr * SL_MULT
                rq = int(CASH * RISK_PCT / sd)
                mq = int(CASH * LEV_CAP / c)
                q = max(min(rq, mq), 1)
                slp = c - sd if sig == "LONG" else c + sd
                tpp = c + sd * rr if sig == "LONG" else c - sd * rr
                pos = {"sym": sym, "d": sig, "e": c, "sl": slp, "tp": tpp,
                       "atr": atr, "q": q, "date": td, "h": t.hour}
                cd_until = ts + cooldown
                pdl_dirs.add(sig)
            pc = c
    return trades_all


# ─── Strategy 2: ORB (Opening Range Breakout) ────────────────────────────

def strat_orb(sym, day_groups, rr):
    """First 15 min (09:15-09:30) defines the range. Trade breakout after."""
    SL_MULT = 1.0
    trades_all = []
    for td, _, today, _, _ in day_groups:
        range_start = time(9, 15)
        range_end = time(9, 30)

        range_bars = today[(today.index.map(lambda x: x.time()) >= range_start) &
                           (today.index.map(lambda x: x.time()) < range_end)]
        if range_bars.empty:
            continue

        orb_high = float(range_bars["high"].max())
        orb_low = float(range_bars["low"].min())
        orb_range = orb_high - orb_low
        if orb_range <= 0:
            continue

        traded_dirs = set()
        pos, dsl, cd_until = None, 0, None
        cooldown = timedelta(minutes=COOLDOWN_MIN)

        active = today[today.index.map(lambda x: x.time()) >= range_end]
        for i in range(len(active)):
            row = active.iloc[i]
            ts = active.index[i]
            t = ts.time()
            c = row["close"]

            if pos is not None:
                hit, pv = None, 0
                if pos["d"] == "LONG":
                    if row["low"] <= pos["sl"]:
                        hit, pv = "SL", (pos["sl"] - pos["e"]) * pos["q"]
                    elif row["high"] >= pos["tp"]:
                        hit, pv = "TP", (pos["tp"] - pos["e"]) * pos["q"]
                else:
                    if row["high"] >= pos["sl"]:
                        hit, pv = "SL", (pos["e"] - pos["sl"]) * pos["q"]
                    elif row["low"] <= pos["tp"]:
                        hit, pv = "TP", (pos["e"] - pos["tp"]) * pos["q"]
                if hit:
                    trades_all.append({**pos, "pnl": pv, "res": hit})
                    if hit == "SL":
                        dsl += 1
                    pos = None
                continue

            if t > ENTRY_END or dsl >= MAX_SL_PER_DAY:
                continue
            if cd_until and ts < cd_until:
                continue

            atr = row["atr"]
            vsma = row["vol_sma"]
            if pd.isna(atr) or atr <= 0 or pd.isna(vsma):
                continue

            vol_ok = row["volume"] > vsma if vsma > 0 else False
            sig = None

            if vol_ok:
                if "LONG" not in traded_dirs and c > orb_high:
                    sig = "LONG"
                elif "SHORT" not in traded_dirs and c < orb_low:
                    sig = "SHORT"

            if sig:
                sd = orb_range * SL_MULT
                sd = max(sd, atr * 0.8)
                rq = int(CASH * RISK_PCT / sd)
                mq = int(CASH * LEV_CAP / c)
                q = max(min(rq, mq), 1)
                slp = c - sd if sig == "LONG" else c + sd
                tpp = c + sd * rr if sig == "LONG" else c - sd * rr
                pos = {"sym": sym, "d": sig, "e": c, "sl": slp, "tp": tpp,
                       "atr": atr, "q": q, "date": td, "h": t.hour}
                traded_dirs.add(sig)
                cd_until = ts + cooldown
    return trades_all


# ─── Strategy 3: VWAP Bounce ─────────────────────────────────────────────

def strat_vwap_bounce(sym, day_groups, rr):
    """Enter when price pulls back to VWAP and bounces with volume confirmation."""
    SL_MULT = 1.2
    trades_all = []
    for td, _, today, _, _ in day_groups:
        pos, dsl, cd_until = None, 0, None
        cooldown = timedelta(minutes=COOLDOWN_MIN)
        prev_above_vwap = None

        for i in range(len(today)):
            row = today.iloc[i]
            ts = today.index[i]
            t = ts.time()
            c = row["close"]
            vwap = row["vwap"]

            if pos is not None:
                hit, pv = None, 0
                if pos["d"] == "LONG":
                    if row["low"] <= pos["sl"]:
                        hit, pv = "SL", (pos["sl"] - pos["e"]) * pos["q"]
                    elif row["high"] >= pos["tp"]:
                        hit, pv = "TP", (pos["tp"] - pos["e"]) * pos["q"]
                else:
                    if row["high"] >= pos["sl"]:
                        hit, pv = "SL", (pos["e"] - pos["sl"]) * pos["q"]
                    elif row["low"] <= pos["tp"]:
                        hit, pv = "TP", (pos["e"] - pos["tp"]) * pos["q"]
                if hit:
                    trades_all.append({**pos, "pnl": pv, "res": hit})
                    if hit == "SL":
                        dsl += 1
                    pos = None
                    prev_above_vwap = c > vwap if not pd.isna(vwap) else None
                continue

            if pd.isna(vwap):
                prev_above_vwap = None
                continue

            if not (ENTRY_START <= t <= ENTRY_END) or dsl >= MAX_SL_PER_DAY:
                prev_above_vwap = c > vwap
                continue
            if cd_until and ts < cd_until:
                prev_above_vwap = c > vwap
                continue

            atr = row["atr"]
            rsi = row["rsi"]
            vsma = row["vol_sma"]
            if pd.isna(atr) or atr <= 0 or pd.isna(rsi) or pd.isna(vsma):
                prev_above_vwap = c > vwap
                continue

            vol_ok = row["volume"] > vsma if vsma > 0 else False
            currently_above = c > vwap
            sig = None

            if prev_above_vwap is not None and vol_ok:
                vwap_dist = abs(c - vwap)
                near_vwap = vwap_dist < atr * 0.5

                if (not prev_above_vwap and currently_above
                        and near_vwap and rsi > 45 and rsi < 65):
                    sig = "LONG"
                elif (prev_above_vwap and not currently_above
                      and near_vwap and rsi > 35 and rsi < 55):
                    sig = "SHORT"

            if sig:
                sd = atr * SL_MULT
                rq = int(CASH * RISK_PCT / sd)
                mq = int(CASH * LEV_CAP / c)
                q = max(min(rq, mq), 1)
                slp = c - sd if sig == "LONG" else c + sd
                tpp = c + sd * rr if sig == "LONG" else c - sd * rr
                pos = {"sym": sym, "d": sig, "e": c, "sl": slp, "tp": tpp,
                       "atr": atr, "q": q, "date": td, "h": t.hour}
                cd_until = ts + cooldown

            prev_above_vwap = currently_above
    return trades_all


# ─── Strategy 4: EMA Scalp ───────────────────────────────────────────────

def strat_ema_scalp(sym, day_groups, rr):
    """EMA-9/21 crossover with volume + RSI confirmation. Quick momentum scalp."""
    SL_MULT = 1.0
    trades_all = []
    for td, _, today, _, _ in day_groups:
        pos, dsl, cd_until = None, 0, None
        cooldown = timedelta(minutes=COOLDOWN_MIN)
        prev_ema_above = None

        for i in range(len(today)):
            row = today.iloc[i]
            ts = today.index[i]
            t = ts.time()
            c = row["close"]

            if pos is not None:
                hit, pv = None, 0
                if pos["d"] == "LONG":
                    if row["low"] <= pos["sl"]:
                        hit, pv = "SL", (pos["sl"] - pos["e"]) * pos["q"]
                    elif row["high"] >= pos["tp"]:
                        hit, pv = "TP", (pos["tp"] - pos["e"]) * pos["q"]
                else:
                    if row["high"] >= pos["sl"]:
                        hit, pv = "SL", (pos["e"] - pos["sl"]) * pos["q"]
                    elif row["low"] <= pos["tp"]:
                        hit, pv = "TP", (pos["e"] - pos["tp"]) * pos["q"]
                if hit:
                    trades_all.append({**pos, "pnl": pv, "res": hit})
                    if hit == "SL":
                        dsl += 1
                    pos = None
                continue

            ema9 = row.get("ema9", np.nan)
            ema21 = row.get("ema21", np.nan)
            if pd.isna(ema9) or pd.isna(ema21):
                prev_ema_above = None
                continue

            currently_above = ema9 > ema21

            if not (ENTRY_START <= t <= ENTRY_END) or dsl >= MAX_SL_PER_DAY:
                prev_ema_above = currently_above
                continue
            if cd_until and ts < cd_until:
                prev_ema_above = currently_above
                continue

            atr = row["atr"]
            rsi = row["rsi"]
            vsma = row["vol_sma"]
            vwap = row["vwap"]
            if pd.isna(atr) or atr <= 0 or pd.isna(rsi) or pd.isna(vsma) or pd.isna(vwap):
                prev_ema_above = currently_above
                continue

            vol_ok = row["volume"] > vsma if vsma > 0 else False
            sig = None

            if prev_ema_above is not None and vol_ok:
                if not prev_ema_above and currently_above and rsi > 50 and c > vwap:
                    sig = "LONG"
                elif prev_ema_above and not currently_above and rsi < 50 and c < vwap:
                    sig = "SHORT"

            if sig:
                sd = atr * SL_MULT
                rq = int(CASH * RISK_PCT / sd)
                mq = int(CASH * LEV_CAP / c)
                q = max(min(rq, mq), 1)
                slp = c - sd if sig == "LONG" else c + sd
                tpp = c + sd * rr if sig == "LONG" else c - sd * rr
                pos = {"sym": sym, "d": sig, "e": c, "sl": slp, "tp": tpp,
                       "atr": atr, "q": q, "date": td, "h": t.hour}
                cd_until = ts + cooldown

            prev_ema_above = currently_above
    return trades_all


# ─── Analysis ────────────────────────────────────────────────────────────

def analyze(trades):
    if not trades:
        return {"n": 0, "wr": 0.0, "pnl": 0.0, "tp": 0, "sl": 0,
                "dd": 0.0, "pf": 0.0, "avg": 0.0}
    tdf = pd.DataFrame(trades)
    n = len(tdf)
    pnl = tdf["pnl"].sum()
    wins = int((tdf["pnl"] > 0).sum())
    tp = int((tdf["res"] == "TP").sum())
    sl = int((tdf["res"] == "SL").sum())
    cumul = tdf["pnl"].cumsum()
    dd = float((cumul - cumul.cummax()).min())
    gw = tdf.loc[tdf["pnl"] > 0, "pnl"].sum()
    gl = abs(tdf.loc[tdf["pnl"] <= 0, "pnl"].sum())
    pf = gw / gl if gl > 0 else (9.99 if gw > 0 else 0.0)
    return {"n": n, "wr": wins / n * 100, "pnl": pnl, "tp": tp, "sl": sl,
            "dd": dd, "pf": min(pf, 9.99), "avg": pnl / n}


# ─── Main ────────────────────────────────────────────────────────────────

STRATEGIES = {
    "PDL-Scaled": {"fn": strat_pdl_scaled, "ind": (27, 42, 60)},
    "ORB":        {"fn": strat_orb,        "ind": (27, 42, 60)},
    "VWAP-Bounce":{"fn": strat_vwap_bounce,"ind": (14, 28, 40)},
    "EMA-Scalp":  {"fn": strat_ema_scalp,  "ind": (14, 28, 40)},
}


def run():
    print("=" * 120)
    print("  MULTI-STRATEGY 1-MIN BACKTEST — Nifty 50 × 90 days")
    print(f"  Period: {START.date()} → {TODAY.date()}")
    print(f"  Strategies: {', '.join(STRATEGIES.keys())}")
    print(f"  RR sweep: {RR_VALUES}")
    print("=" * 120)

    # 1. Fetch data
    print("\n[1] Fetching 1-min data ...")
    fetch_all_1min()

    # 2. Load and precompute per strategy
    print("[2] Loading data & computing indicators per strategy ...\n")

    stock_cache = {}
    for sym, sec in NIFTY50:
        p = csv_1min(sym)
        if not p.exists():
            continue
        try:
            df = load_csv(str(p))
            df = df[df.index >= START - pd.Timedelta(days=10)]
            if len(df) < 200:
                continue
            stock_cache[sym] = {"raw": df, "sector": sec}
        except Exception:
            continue

    print(f"  {len(stock_cache)} stocks loaded\n")

    strat_data = {}
    for sname, sdef in STRATEGIES.items():
        rsi_p, atr_p, vol_p = sdef["ind"]
        strat_data[sname] = {}
        for sym, info in stock_cache.items():
            ind = compute_indicators(info["raw"], rsi_p, atr_p, vol_p)
            ind = ind[ind.index >= START]
            if len(ind) < 100:
                continue
            dg = get_day_groups(ind)
            if len(dg) >= 5:
                strat_data[sname][sym] = dg
        print(f"  {sname:14s}: {len(strat_data[sname])} stocks ready")

    # 3. Run all strategies × RR sweep
    print(f"\n[3] Running strategy × RR sweep ...\n")

    master_results = []

    for sname, sdef in STRATEGIES.items():
        fn = sdef["fn"]
        for rr in RR_VALUES:
            all_trades = []
            stock_pnl = {}
            for sym in strat_data[sname]:
                dg = strat_data[sname][sym]
                t = fn(sym, dg, rr)
                all_trades.extend(t)
                stock_pnl[sym] = sum(x["pnl"] for x in t)

            a = analyze(all_trades)
            days_active = len(set(t["date"] for t in all_trades)) if all_trades else 1
            master_results.append({
                "strategy": sname, "rr": rr, **a,
                "daily": a["pnl"] / max(days_active, 1),
                "stock_pnl": stock_pnl,
            })
            print(f"  {sname:14s}  RR={rr:.1f}  →  {a['n']:>4} trades  "
                  f"WR={a['wr']:>5.1f}%  P&L=₹{a['pnl']:>9,.0f}  "
                  f"DD=₹{a['dd']:>7,.0f}  PF={a['pf']:.2f}")
        print()

    # 4. Grand comparison — best RR per strategy
    print("\n" + "=" * 120)
    print("  GRAND COMPARISON — Best RR per Strategy")
    print("=" * 120)

    best_per_strat = {}
    for sname in STRATEGIES:
        strat_rows = [r for r in master_results if r["strategy"] == sname]
        best = max(strat_rows, key=lambda x: x["pnl"])
        best_per_strat[sname] = best

    print(f"  {'Strategy':<14} {'BestRR':>6} │ {'Trades':>6} {'WR%':>6} │ "
          f"{'P&L':>10} {'Avg/Tr':>9} {'Daily':>9} │ {'TP':>4} {'SL':>4} │ "
          f"{'MaxDD':>9} {'PF':>5}")
    print("  " + "─" * 105)

    overall_best = max(best_per_strat.values(), key=lambda x: x["pnl"])
    for sname in STRATEGIES:
        r = best_per_strat[sname]
        marker = "  ◀ WINNER" if r is overall_best else ""
        print(f"  {sname:<14} {r['rr']:>5.1f}x │ {r['n']:>6} {r['wr']:>5.1f}% │ "
              f"₹{r['pnl']:>8,.0f} ₹{r['avg']:>7,.0f} ₹{r['daily']:>7,.0f} │ "
              f"{r['tp']:>4} {r['sl']:>4} │ ₹{r['dd']:>7,.0f} {r['pf']:>5.2f}{marker}")

    # 5. Top 10 stocks for the winning strategy
    winner = overall_best
    print(f"\n{'=' * 120}")
    print(f"  TOP 10 STOCKS — {winner['strategy']} (RR={winner['rr']})")
    print("=" * 120)

    sp = winner["stock_pnl"]
    ranked = sorted(sp.items(), key=lambda x: x[1], reverse=True)

    print(f"  {'#':>3} {'Stock':<14} {'Sector':<12} {'P&L':>10}")
    print("  " + "─" * 45)
    for i, (sym, pnl) in enumerate(ranked[:10], 1):
        sec = stock_cache.get(sym, {}).get("sector", "?")
        print(f"  {i:>3} {sym:<14} {sec:<12} ₹{pnl:>8,.0f} ★")

    print(f"\n  Bottom 5:")
    for sym, pnl in ranked[-5:]:
        sec = stock_cache.get(sym, {}).get("sector", "?")
        print(f"      {sym:<14} {sec:<12} ₹{pnl:>8,.0f}")

    # 6. Also show top 10 for each strategy
    for sname in STRATEGIES:
        if sname == winner["strategy"]:
            continue
        r = best_per_strat[sname]
        sp2 = r["stock_pnl"]
        ranked2 = sorted(sp2.items(), key=lambda x: x[1], reverse=True)
        print(f"\n  Top 10 — {sname} (RR={r['rr']}):")
        for i, (sym, pnl) in enumerate(ranked2[:10], 1):
            sec = stock_cache.get(sym, {}).get("sector", "?")
            print(f"    {i:>2}. {sym:<14} {sec:<12} ₹{pnl:>8,.0f}")

    # 7. Compare with current 3-min PDL
    print(f"\n{'=' * 120}")
    print("  COMPARISON: Best 1-min Strategy vs Current 3-min PDL (RR=2.5)")
    print("=" * 120)
    print(f"  Current 3-min PDL: ~₹59,122 P&L over 90 days (top 10 stocks)")
    print(f"  Best 1-min {winner['strategy']}: ₹{winner['pnl']:,.0f} P&L ({len(sp)} stocks)")

    top10_pnl = sum(pnl for _, pnl in ranked[:10])
    print(f"  Best 1-min top-10 only: ₹{top10_pnl:,.0f}")
    diff = top10_pnl - 59122
    print(f"  Difference: ₹{diff:+,.0f}")
    if diff > 0:
        print(f"  → 1-min {winner['strategy']} BEATS 3-min PDL by ₹{diff:,.0f}")
    else:
        print(f"  → 3-min PDL still wins by ₹{abs(diff):,.0f}")

    print("=" * 120)
    print("\n  STOCKS config (winner):")
    print("  STOCKS = [")
    for sym, pnl in ranked[:10]:
        sec = stock_cache.get(sym, {}).get("sector", "?")
        print(f'      ("{sym}", "{sec}"),')
    print("  ]")
    print("=" * 120)


if __name__ == "__main__":
    run()

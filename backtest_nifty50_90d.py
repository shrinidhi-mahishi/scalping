"""
Full Nifty 50 backtest — previous 90 days.
Uses current live_signals.py strategy. Fetches missing data via Angel One API.
Ranks all stocks by P&L and picks top 10.

Covers tasks:
  - Backtest 90 days on all Nifty 50 stocks
  - Pick top 10 stocks by P&L

Usage:
    python backtest_nifty50_90d.py
"""

import sys
import time as _time
from datetime import time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(".").resolve()))
from fetch_data import AngelOneClient, load_csv, save_csv
from live_signals import (
    RR, RISK_PCT, LEV_CAP,
    PDL_SL_MULT, MOM_SL_MULT,
    MOM_VOL_MULT, MOM_BODY_RATIO,
    MOM_LONG_RSI, MOM_SHORT_RSI,
    PDL_LONG_RSI_MIN, PDL_SHORT_RSI_MAX,
    MAX_SL_PER_DAY,
    STAG_MAX_BARS, STAG_MIN_PROFIT_ATR,
    ENTRY_AM, ENTRY_PM, COOLDOWN,
    compute_indicators,
)

CASH = 50_000.0
TODAY = pd.Timestamp.now().normalize()
START = TODAY - pd.Timedelta(days=90)
_F = str.maketrans({"&": ""})

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


def csv_path(sym):
    fname = sym.translate(_F)
    p = Path("data") / f"{fname}_3min.csv"
    if p.exists():
        return p
    p2 = Path("data") / f"{fname.lower()}_3min.csv"
    if p2.exists():
        return p2
    return Path("data") / f"{fname}_3min.csv"


def fetch_all_missing():
    needs = []
    for sym, _ in NIFTY50:
        p = csv_path(sym)
        if not p.exists():
            needs.append((sym, (TODAY - timedelta(days=7)).strftime("%Y-%m-%d"), TODAY.strftime("%Y-%m-%d")))
            continue
        df = load_csv(str(p))
        last = df.index.max().date()
        if last < TODAY.date():
            fr = (last + timedelta(days=1)).strftime("%Y-%m-%d")
            needs.append((sym, fr, TODAY.strftime("%Y-%m-%d")))

    if not needs:
        print("[data] All 48 stocks up to date.\n")
        return

    print(f"[data] {len(needs)} stocks need update. Fetching from Angel One...\n")
    client = AngelOneClient()
    client.connect()

    for i, (sym, fr, to) in enumerate(needs):
        p = csv_path(sym)
        print(f"  [{i+1}/{len(needs)}] {sym:14s} {fr} → {to} ... ", end="", flush=True)
        try:
            new = client.fetch_candles(sym, f"{fr} 09:15", f"{to} 15:30")
            if p.exists():
                old = load_csv(str(p))
                combined = pd.concat([old, new])
                combined = combined[~combined.index.duplicated(keep="last")].sort_index()
            else:
                combined = new
            save_csv(combined, str(p))
            print(f"{len(new)} bars")
            _time.sleep(0.5)
        except Exception as e:
            print(f"FAILED: {e}")
            _time.sleep(1)

    print()


def precompute(sym):
    p = csv_path(sym)
    if not p.exists():
        return None
    try:
        df = load_csv(str(p))
        ind = compute_indicators(df)
        ind = ind[ind.index >= START]
        if len(ind) < 50:
            return None
        days = sorted(ind.index.normalize().unique())
        out = []
        for i in range(1, len(days)):
            td, pd_ = days[i], days[i - 1]
            pr = ind[ind.index.normalize() == pd_]
            tr = ind[ind.index.normalize() == td]
            if pr.empty or tr.empty:
                continue
            out.append((td, float(pr["high"].max()), float(pr["low"].min()), tr, pr))
        return out
    except Exception as e:
        return None


def in_entry_window(t):
    if ENTRY_AM and ENTRY_AM[0] <= t <= ENTRY_AM[1]:
        return True
    if ENTRY_PM and ENTRY_PM[0] <= t <= ENTRY_PM[1]:
        return True
    return False


def simulate(day_data, sym):
    trades = []
    for today, pdl_h, pdl_l, today_rows, _ in day_data:
        pdl_dirs, cd_until, pos, bh, pc, dsl = set(), None, None, 0, None, 0
        for i in range(len(today_rows)):
            row = today_rows.iloc[i]
            ts = today_rows.index[i]
            t = ts.time()
            c = row["close"]

            if pos is not None:
                bh += 1
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
                    pos, bh, pc = None, 0, c
                    continue
                if bh >= STAG_MAX_BARS:
                    u = (c - pos["e"]) if pos["d"] == "LONG" else (pos["e"] - c)
                    if u < STAG_MIN_PROFIT_ATR * pos["atr"]:
                        trades.append({**pos, "pnl": u * pos["q"], "res": "STAG"})
                        pos, bh = None, 0
                pc = c
                continue

            if not in_entry_window(t) or (cd_until and ts < cd_until) or dsl >= MAX_SL_PER_DAY:
                pc = c
                continue

            rsi, atr, vwap, vsma = row["rsi"], row["atr"], row["vwap"], row["vol_sma"]
            if pd.isna(rsi) or pd.isna(atr) or atr <= 0 or pd.isna(vwap) or pd.isna(vsma):
                pc = c
                continue

            vol_ok = row["volume"] > vsma if vsma > 0 else False
            sig, trig = None, None

            if pc is not None and vol_ok:
                if ("LONG" not in pdl_dirs and pc <= pdl_h and c > pdl_h
                        and c > vwap and rsi > PDL_LONG_RSI_MIN):
                    sig, trig = "LONG", "PDL"
                elif ("SHORT" not in pdl_dirs and pc >= pdl_l and c < pdl_l
                      and c < vwap and rsi < PDL_SHORT_RSI_MAX):
                    sig, trig = "SHORT", "PDL"

            if sig is None:
                br = row.get("body_ratio", 0)
                vr = row.get("vol_ratio", 0)
                if not pd.isna(br) and not pd.isna(vr):
                    if vr >= MOM_VOL_MULT and br >= MOM_BODY_RATIO:
                        if c > row["open"] and c > vwap and MOM_LONG_RSI[0] <= rsi <= MOM_LONG_RSI[1]:
                            sig, trig = "LONG", "MOM"
                        elif c < row["open"] and c < vwap and MOM_SHORT_RSI[0] <= rsi <= MOM_SHORT_RSI[1]:
                            sig, trig = "SHORT", "MOM"

            if sig:
                sm = PDL_SL_MULT if trig == "PDL" else MOM_SL_MULT
                sd = atr * sm
                rq = int(CASH * RISK_PCT / sd)
                mq = int(CASH * LEV_CAP / c)
                q = max(min(rq, mq), 1)
                slp = c - sd if sig == "LONG" else c + sd
                tpp = c + sd * RR if sig == "LONG" else c - sd * RR
                pos = {
                    "sym": sym, "d": sig, "trig": trig, "e": c, "sl": slp,
                    "tp": tpp, "atr": atr, "q": q, "h": t.hour, "date": today,
                }
                bh = 0
                cd_until = ts + COOLDOWN
                if trig == "PDL":
                    pdl_dirs.add(sig)
            pc = c
    return trades


def analyze(trades):
    if not trades:
        return {"n": 0, "wins": 0, "wr": 0.0, "pnl": 0.0, "tp": 0, "sl": 0,
                "stag": 0, "dd": 0.0, "avg": 0.0, "pf": 0.0}
    tdf = pd.DataFrame(trades)
    n = len(tdf)
    pnl = tdf["pnl"].sum()
    wins = int((tdf["pnl"] > 0).sum())
    tp = int((tdf["res"] == "TP").sum())
    sl = int((tdf["res"] == "SL").sum())
    stag = int((tdf["res"] == "STAG").sum())
    cumul = tdf["pnl"].cumsum()
    dd = float((cumul - cumul.cummax()).min())
    gw = tdf.loc[tdf["pnl"] > 0, "pnl"].sum()
    gl = abs(tdf.loc[tdf["pnl"] <= 0, "pnl"].sum())
    pf = gw / gl if gl > 0 else (9.99 if gw > 0 else 0.0)
    return {"n": n, "wins": wins, "wr": wins / n * 100, "pnl": pnl,
            "tp": tp, "sl": sl, "stag": stag, "dd": dd,
            "avg": pnl / n, "pf": min(pf, 9.99)}


def run():
    print("=" * 120)
    print(f"  NIFTY 50 BACKTEST — 90 DAYS ({START.date()} to {TODAY.date()})")
    print(f"  Strategy: Morning-Only (10-12), MAX_SL={MAX_SL_PER_DAY}/stock/day")
    print(f"  RR={RR}, PDL_SL={PDL_SL_MULT}, MOM_SL={MOM_SL_MULT}, Capital=₹{CASH:,.0f}")
    print("=" * 120)

    # 1. Fetch missing data
    fetch_all_missing()

    # 2. Precompute
    print("Loading & computing indicators...")
    precomputed = {}
    failed = []
    for sym, sec in NIFTY50:
        data = precompute(sym)
        if data and len(data) >= 5:
            precomputed[sym] = data
        else:
            failed.append(sym)

    print(f"  {len(precomputed)}/{len(NIFTY50)} stocks loaded")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    print()

    # 3. Run backtest
    all_trades = []
    stock_results = []

    for sym, sec in NIFTY50:
        if sym not in precomputed:
            continue
        trades = simulate(precomputed[sym], sym)
        all_trades.extend(trades)
        s = analyze(trades)
        stock_results.append({"sym": sym, "sector": sec, **s})

    agg = analyze(all_trades)

    # 4. Aggregate
    print("=" * 120)
    print("  AGGREGATE — ALL STOCKS")
    print("=" * 120)
    print(f"  Trades: {agg['n']}  |  Win Rate: {agg['wr']:.1f}%  |  P&L: ₹{agg['pnl']:,.0f}")
    print(f"  TP: {agg['tp']}  |  SL: {agg['sl']}  |  Stag: {agg['stag']}")
    print(f"  Max DD: ₹{agg['dd']:,.0f}  |  PF: {agg['pf']:.2f}  |  Avg/Trade: ₹{agg['avg']:,.0f}")

    # 5. All stocks ranked
    stock_results.sort(key=lambda x: x["pnl"], reverse=True)

    print()
    print("=" * 120)
    print("  ALL STOCKS RANKED BY P&L")
    print("=" * 120)
    print(
        f"  {'#':>3} {'Stock':<14} {'Sector':<12} │ {'Trades':>6} {'Win%':>6} "
        f"{'P&L':>10} │ {'TP':>4} {'SL':>4} {'Stag':>4} │ {'MaxDD':>9} {'PF':>5} {'Avg':>7}"
    )
    print("  " + "─" * 110)

    for i, s in enumerate(stock_results, 1):
        marker = " ★" if i <= 10 else ""
        print(
            f"  {i:>3} {s['sym']:<14} {s['sector']:<12} │ {s['n']:>6} {s['wr']:>5.1f}% "
            f"₹{s['pnl']:>8,.0f} │ {s['tp']:>4} {s['sl']:>4} {s['stag']:>4} "
            f"│ ₹{s['dd']:>7,.0f} {s['pf']:>5.2f} ₹{s['avg']:>5,.0f}{marker}"
        )

    # 6. Top 10 summary
    top10 = stock_results[:10]
    top10_trades = [t for t in all_trades if t["sym"] in {s["sym"] for s in top10}]
    top10_agg = analyze(top10_trades)

    print()
    print("=" * 120)
    print("  TOP 10 STOCKS — COMBINED PERFORMANCE")
    print("=" * 120)
    print(f"  Trades: {top10_agg['n']}  |  Win Rate: {top10_agg['wr']:.1f}%  |  P&L: ₹{top10_agg['pnl']:,.0f}")
    print(f"  TP: {top10_agg['tp']}  |  SL: {top10_agg['sl']}  |  Stag: {top10_agg['stag']}")
    print(f"  Max DD: ₹{top10_agg['dd']:,.0f}  |  PF: {top10_agg['pf']:.2f}  |  Avg/Trade: ₹{top10_agg['avg']:,.0f}")

    # Sector distribution
    sectors = {}
    for s in top10:
        sectors[s["sector"]] = sectors.get(s["sector"], 0) + 1
    print(f"\n  Sector mix: {', '.join(f'{k}({v})' for k, v in sorted(sectors.items(), key=lambda x: -x[1]))}")

    # Config output
    print(f"\n  STOCKS config for live_signals.py:")
    print("  STOCKS = [")
    for s in top10:
        print(f'      ("{s["sym"]}", "{s["sector"]}"),')
    print("  ]")

    # Bottom 10
    bottom10 = stock_results[-10:]
    print()
    print("=" * 120)
    print("  BOTTOM 10 — WORST PERFORMERS (AVOID)")
    print("=" * 120)
    for s in bottom10:
        print(f"  {s['sym']:<14} {s['sector']:<12} ₹{s['pnl']:>8,.0f}  WR {s['wr']:.1f}%  DD ₹{s['dd']:,.0f}")

    # Compare with current STOCKS
    from live_signals import STOCKS as CURRENT_STOCKS
    current_syms = {s for s, _ in CURRENT_STOCKS}
    new_syms = {s["sym"] for s in top10}
    added = new_syms - current_syms
    removed = current_syms - new_syms
    kept = current_syms & new_syms

    print()
    print("=" * 120)
    print("  COMPARISON WITH CURRENT live_signals.py STOCKS")
    print("=" * 120)
    print(f"  Kept ({len(kept)}):    {', '.join(sorted(kept))}")
    print(f"  Added ({len(added)}):   {', '.join(sorted(added)) if added else 'none'}")
    print(f"  Removed ({len(removed)}): {', '.join(sorted(removed)) if removed else 'none'}")
    print("=" * 120)


if __name__ == "__main__":
    run()

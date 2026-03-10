"""Quick RR sweep — find optimal reward-to-risk ratio for PDL strategy."""

import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(".").resolve()))
from fetch_data import load_csv
from live_signals import (
    STOCKS, RISK_PCT, LEV_CAP,
    PDL_SL_MULT, PDL_LONG_RSI_MIN, PDL_SHORT_RSI_MAX,
    MAX_SL_PER_DAY, ENTRY_AM, ENTRY_PM, COOLDOWN,
    compute_indicators,
)

CASH = 50_000.0
TODAY = pd.Timestamp.now().normalize()
START = TODAY - pd.Timedelta(days=90)
_F = str.maketrans({"&": ""})

RR_VALUES = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0]


def resample_to_3min(df):
    return df.resample("3min").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna(subset=["open"])


def precompute(df):
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
        out.append((td, float(pr["high"].max()), float(pr["low"].min()), tr))
    return out


def in_entry(t):
    if ENTRY_AM and ENTRY_AM[0] <= t <= ENTRY_AM[1]:
        return True
    if ENTRY_PM and ENTRY_PM[0] <= t <= ENTRY_PM[1]:
        return True
    return False


def simulate(day_data, sym, rr):
    trades = []
    for today, pdl_h, pdl_l, rows in day_data:
        pdl_dirs, cd_until, pos, pc, dsl = set(), None, None, None, 0
        for i in range(len(rows)):
            row = rows.iloc[i]
            ts = rows.index[i]
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
                    trades.append({**pos, "pnl": pv, "res": hit})
                    if hit == "SL":
                        dsl += 1
                    pos, pc = None, c
                    continue
                pc = c
                continue

            if not in_entry(t) or (cd_until and ts < cd_until) or dsl >= MAX_SL_PER_DAY:
                pc = c
                continue

            rsi, atr, vwap, vsma = row["rsi"], row["atr"], row["vwap"], row["vol_sma"]
            if pd.isna(rsi) or pd.isna(atr) or atr <= 0 or pd.isna(vwap) or pd.isna(vsma):
                pc = c
                continue

            vol_ok = row["volume"] > vsma if vsma > 0 else False
            sig = None
            if pc is not None and vol_ok:
                if ("LONG" not in pdl_dirs and pc <= pdl_h and c > pdl_h
                        and c > vwap and rsi > PDL_LONG_RSI_MIN):
                    sig = "LONG"
                elif ("SHORT" not in pdl_dirs and pc >= pdl_l and c < pdl_l
                      and c < vwap and rsi < PDL_SHORT_RSI_MAX):
                    sig = "SHORT"

            if sig:
                sd = atr * PDL_SL_MULT
                rq = int(CASH * RISK_PCT / sd)
                mq = int(CASH * LEV_CAP / c)
                q = max(min(rq, mq), 1)
                slp = c - sd if sig == "LONG" else c + sd
                tpp = c + sd * rr if sig == "LONG" else c - sd * rr
                pos = {"sym": sym, "d": sig, "e": c, "sl": slp, "tp": tpp,
                       "atr": atr, "q": q, "date": today}
                cd_until = ts + COOLDOWN
                pdl_dirs.add(sig)
            pc = c
    return trades


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


print("=" * 110)
print("  RR SWEEP — PDL Strategy on Top 10 Stocks (90 days)")
print(f"  Period: {START.date()} to {TODAY.date()}")
print(f"  SL={PDL_SL_MULT}×ATR, CD={int(COOLDOWN.total_seconds()//60)}min")
print("=" * 110)

# Load data (resample 1-min → 3-min)
stock_data = {}
for sym, _ in STOCKS:
    p = Path("data") / f"{sym.translate(_F)}_1min.csv"
    if not p.exists():
        continue
    df = load_csv(str(p))
    df3 = resample_to_3min(df)
    dd = precompute(df3)
    if dd:
        stock_data[sym] = dd
        print(f"  {sym:14s} loaded ({len(dd)} days)")

print(f"\n  {len(stock_data)} stocks ready\n")

# Sweep
results = []
for rr in RR_VALUES:
    all_trades = []
    for sym, dd in stock_data.items():
        all_trades.extend(simulate(dd, sym, rr))
    a = analyze(all_trades)
    days = len(set(t["date"] for t in all_trades)) if all_trades else 1
    a["daily"] = a["pnl"] / max(days, 1)
    results.append({"rr": rr, **a})

print("=" * 110)
print(f"  {'RR':>5} │ {'Trades':>6} {'WR%':>6} │ {'P&L':>10} {'Avg/Tr':>9} {'Daily':>9} │ "
      f"{'TP':>4} {'SL':>4} │ {'MaxDD':>9} {'PF':>5}")
print("  " + "─" * 100)

best = max(results, key=lambda x: x["pnl"])
for r in results:
    marker = "  ◀ BEST" if r["rr"] == best["rr"] else ""
    marker2 = "  ◀ CURRENT" if r["rr"] == 2.5 else marker
    print(f"  {r['rr']:>5.2f} │ {r['n']:>6} {r['wr']:>5.1f}% │ "
          f"₹{r['pnl']:>8,.0f} ₹{r['avg']:>7,.0f} ₹{r['daily']:>7,.0f} │ "
          f"{r['tp']:>4} {r['sl']:>4} │ ₹{r['dd']:>7,.0f} {r['pf']:>5.2f}{marker2}")

print("=" * 110)

# Show improvement
if best["rr"] != 2.5:
    curr = next(r for r in results if r["rr"] == 2.5)
    diff = best["pnl"] - curr["pnl"]
    print(f"\n  RECOMMENDATION: Switch RR from 2.5 → {best['rr']}")
    print(f"  P&L improvement: ₹{diff:+,.0f} ({diff/abs(curr['pnl'])*100:+.0f}%)")
    print(f"  Win rate: {curr['wr']:.1f}% → {best['wr']:.1f}%")
    print(f"  Drawdown: ₹{curr['dd']:,.0f} → ₹{best['dd']:,.0f}")
else:
    print(f"\n  Current RR=2.5 is already optimal.")

# Per-stock breakdown for current vs best
if best["rr"] != 2.5:
    print(f"\n  PER-STOCK: RR={best['rr']} vs RR=2.5")
    print(f"  {'Stock':<14} │ {'RR=2.5 P&L':>10} │ {'RR='+str(best['rr'])+' P&L':>12} │ {'Diff':>10}")
    print("  " + "─" * 60)
    for sym, _ in STOCKS:
        if sym not in stock_data:
            continue
        t_curr = simulate(stock_data[sym], sym, 2.5)
        t_best = simulate(stock_data[sym], sym, best["rr"])
        p_curr = sum(t["pnl"] for t in t_curr)
        p_best = sum(t["pnl"] for t in t_best)
        dp = p_best - p_curr
        sign = "+" if dp >= 0 else ""
        print(f"  {sym:<14} │ ₹{p_curr:>8,.0f} │ ₹{p_best:>10,.0f} │ {sign}₹{abs(dp):>7,.0f}")

print("=" * 110)

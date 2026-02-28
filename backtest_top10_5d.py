"""
Backtest top 10 stocks (from live_signals.py) for previous 5 trading days.
Uses the exact strategy from live_signals.py.
Includes day-by-day breakdown and full trade log.

Usage:
    python backtest_top10_5d.py
"""

import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(".").resolve()))
from fetch_data import load_csv
from live_signals import (
    STOCKS, RR, RISK_PCT, LEV_CAP,
    PDL_SL_MULT, MOM_SL_MULT, MOM_VOL_MULT, MOM_BODY_RATIO,
    MOM_LONG_RSI, MOM_SHORT_RSI, PDL_LONG_RSI_MIN, PDL_SHORT_RSI_MAX,
    MAX_SL_PER_DAY, STAG_MAX_BARS, STAG_MIN_PROFIT_ATR,
    ENTRY_AM, ENTRY_PM, COOLDOWN, compute_indicators,
)

CASH = 50_000.0
TODAY = pd.Timestamp.now().normalize()
_F = str.maketrans({"&": ""})


def csv_path(sym):
    fname = sym.translate(_F)
    p = Path("data") / f"{fname}_3min.csv"
    return p if p.exists() else Path("data") / f"{fname.lower()}_3min.csv"


def precompute(sym):
    p = csv_path(sym)
    if not p.exists():
        return None
    try:
        df = load_csv(str(p))
        ind = compute_indicators(df)
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
    except Exception:
        return None


def get_last_n_days(day_data, n=5):
    """Return only the last N trading days from precomputed data."""
    if not day_data:
        return []
    all_dates = sorted(set(d for d, _, _, _, _ in day_data))
    last_n = all_dates[-n:] if len(all_dates) >= n else all_dates
    cutoff = last_n[0]
    return [(d, h, l, r, p) for d, h, l, r, p in day_data if d >= cutoff]


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
                    trades.append({**pos, "pnl": pv, "res": hit, "et": ts})
                    if hit == "SL":
                        dsl += 1
                    pos, bh, pc = None, 0, c
                    continue
                if bh >= STAG_MAX_BARS:
                    u = (c - pos["e"]) if pos["d"] == "LONG" else (pos["e"] - c)
                    if u < STAG_MIN_PROFIT_ATR * pos["atr"]:
                        trades.append({**pos, "pnl": u * pos["q"], "res": "STAG", "et": ts})
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
                    "etime": ts,
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
    precomputed = {}
    date_range = ""
    for sym, _ in STOCKS:
        data = precompute(sym)
        if data:
            last5 = get_last_n_days(data, 5)
            if last5:
                precomputed[sym] = last5
                if not date_range:
                    dates = sorted(set(d.date() for d, _, _, _, _ in last5))
                    date_range = f"{dates[0]} to {dates[-1]}"

    print("=" * 110)
    print(f"  TOP 10 BACKTEST — LAST 5 TRADING DAYS ({date_range})")
    print(f"  Strategy: {ENTRY_AM}, MAX_SL={MAX_SL_PER_DAY}/stock/day, RR={RR}")
    print(f"  Stocks: {', '.join(s for s, _ in STOCKS)}")
    print("=" * 110)
    print(f"  {len(precomputed)}/{len(STOCKS)} stocks loaded\n")

    all_trades = []
    sym_trades = {}
    for sym, _ in STOCKS:
        if sym not in precomputed:
            continue
        t = simulate(precomputed[sym], sym)
        all_trades.extend(t)
        sym_trades[sym] = t

    agg = analyze(all_trades)

    # Aggregate
    print("=" * 110)
    print("  AGGREGATE")
    print("=" * 110)
    print(f"  Trades: {agg['n']}  |  WR: {agg['wr']:.1f}%  |  P&L: ₹{agg['pnl']:,.0f}")
    print(f"  TP: {agg['tp']}  |  SL: {agg['sl']}  |  Stag: {agg['stag']}")
    print(f"  MaxDD: ₹{agg['dd']:,.0f}  |  PF: {agg['pf']:.2f}  |  Avg/Tr: ₹{agg['avg']:,.0f}")

    # Per stock
    print()
    print(f"  {'Stock':<14} │ {'Tr':>4} {'Win%':>6} {'TP':>3} {'SL':>3} {'Stag':>4} │ {'P&L':>9} {'Avg':>7}")
    print("  " + "─" * 65)
    sorted_syms = sorted(sym_trades.keys(), key=lambda s: sum(t["pnl"] for t in sym_trades[s]), reverse=True)
    for sym in sorted_syms:
        s = analyze(sym_trades[sym])
        print(
            f"  {sym:<14} │ {s['n']:>4} {s['wr']:>5.1f}% {s['tp']:>3} {s['sl']:>3} {s['stag']:>4} "
            f"│ ₹{s['pnl']:>7,.0f} ₹{s['avg']:>5,.0f}"
        )
    print("  " + "─" * 65)
    print(f"  {'TOTAL':<14} │ {agg['n']:>4} {agg['wr']:>5.1f}%{' ':>11} │ ₹{agg['pnl']:>7,.0f}")

    # Day by day
    if all_trades:
        tdf = pd.DataFrame(all_trades)
        print()
        print(f"  {'Day':<14} │ {'Tr':>4} {'Win':>4} {'P&L':>9} │ {'Best':>14} {'Worst':>14}")
        print("  " + "─" * 70)
        for dt in sorted(tdf["date"].unique()):
            day = tdf[tdf["date"] == dt]
            dp = day["pnl"].sum()
            dn = len(day)
            dw = int((day["pnl"] > 0).sum())
            bs = day.groupby("sym")["pnl"].sum()
            print(
                f"  {pd.Timestamp(dt).strftime('%a %b %d'):<14} │ {dn:>4} {dw:>4} ₹{dp:>7,.0f} "
                f"│ {bs.idxmax():>14} {bs.idxmin():>14}"
            )

        # Trade log
        print()
        print(f"  {'Time':<14} {'Stock':<12} {'Dir':<6} {'Trig':<5} {'Entry':>9} {'Res':<5} {'P&L':>8}")
        print("  " + "─" * 70)
        for t in sorted(all_trades, key=lambda x: x["etime"]):
            print(
                f"  {t['etime'].strftime('%m-%d %H:%M'):<14} {t['sym']:<12} {t['d']:<6} {t['trig']:<5} "
                f"₹{t['e']:>7,.1f} {t['res']:<5} ₹{t['pnl']:>6,.0f}"
            )

    print("=" * 110)


if __name__ == "__main__":
    run()

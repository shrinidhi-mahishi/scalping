"""
Backtest window-close sweep: compare 09:30-13:00, 09:30-14:00, 09:30-14:30
for both 30-day and 90-day periods on the current top-10 stocks.
"""

import sys
from datetime import time, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(".").resolve()))
from fetch_data import load_csv
from live_signals import (
    STOCKS, RR, RISK_PCT, LEV_CAP,
    PDL_PREARM_BUFFER_ATR, PDL_SL_MULT, MAX_SL_PER_DAY, COOLDOWN,
    compute_indicators,
)

CASH = 15_000.0
TODAY = pd.Timestamp.now().normalize()

WINDOWS = [
    ("09:30–13:00", time(9, 30), time(13, 0)),
    ("09:30–14:00", time(9, 30), time(14, 0)),
    ("09:30–14:30", time(9, 30), time(14, 30)),
]

PERIODS = [
    ("30D", 30),
    ("90D", 90),
]

_F = str.maketrans({"&": ""})


def csv_path(sym):
    fname = sym.translate(_F)
    p = Path("data") / f"{fname}_3min.csv"
    return p if p.exists() else Path("data") / f"{fname.lower()}_3min.csv"


def csv_path_1min(sym):
    fname = sym.translate(_F)
    return Path("data") / f"{fname}_1min.csv"


def resample_3min(df):
    return df.resample("3min").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna(subset=["open"])


def load_symbol_df(sym):
    p1 = csv_path_1min(sym)
    if p1.exists():
        return resample_3min(load_csv(str(p1)))
    p3 = csv_path(sym)
    if p3.exists():
        return load_csv(str(p3))
    return None


def precompute(sym, start):
    df = load_symbol_df(sym)
    if df is None:
        return None
    try:
        ind = compute_indicators(df)
        ind = ind[ind.index >= start]
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
    except Exception:
        return None


def simulate(day_data, sym, win_start, win_end):
    trades = []
    for today, pdl_h, pdl_l, today_rows, _ in day_data:
        pdl_dirs, cd_until, pos, activate_idx, dsl = set(), None, None, None, 0
        for i in range(len(today_rows)):
            row = today_rows.iloc[i]
            ts = today_rows.index[i]
            t = ts.time()
            c = float(row["close"])

            if pos is not None:
                if activate_idx is not None and i >= activate_idx:
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
                        pos, activate_idx = None, None
                continue

            if not (win_start <= t <= win_end):
                continue
            if cd_until and ts < cd_until:
                continue
            if dsl >= MAX_SL_PER_DAY:
                continue

            atr = row["atr"]
            prev_close = float(today_rows.iloc[i - 1]["close"]) if i > 0 else None
            if prev_close is None or pd.isna(atr) or atr <= 0:
                continue

            buffer = float(atr) * PDL_PREARM_BUFFER_ATR
            sig, entry = None, None
            if ("LONG" not in pdl_dirs
                    and prev_close <= pdl_h
                    and row["high"] >= pdl_h + buffer
                    and c >= pdl_h):
                sig, entry = "LONG", pdl_h + buffer
            elif ("SHORT" not in pdl_dirs
                  and prev_close >= pdl_l
                  and row["low"] <= pdl_l - buffer
                  and c <= pdl_l):
                sig, entry = "SHORT", pdl_l - buffer

            if sig:
                sd = float(atr) * PDL_SL_MULT
                rq = int(CASH * RISK_PCT / sd)
                mq = int(CASH * LEV_CAP / entry)
                q = max(min(rq, mq), 1)
                slp = entry - sd if sig == "LONG" else entry + sd
                tpp = entry + sd * RR if sig == "LONG" else entry - sd * RR
                pos = {
                    "sym": sym, "d": sig, "trig": "PDL", "e": entry, "sl": slp,
                    "tp": tpp, "atr": float(atr), "q": q, "h": t.hour, "date": today,
                    "etime": ts,
                }
                cd_until = ts + COOLDOWN
                activate_idx = i + 1
                pdl_dirs.add(sig)
    return trades


def analyze(trades):
    if not trades:
        return {"n": 0, "wins": 0, "wr": 0.0, "pnl": 0.0, "tp": 0, "sl": 0,
                "dd": 0.0, "avg": 0.0, "pf": 0.0}
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
    return {"n": n, "wins": wins, "wr": wins / n * 100, "pnl": pnl,
            "tp": tp, "sl": sl, "dd": dd, "avg": pnl / n, "pf": min(pf, 9.99)}


def run():
    stock_names = [s for s, _ in STOCKS]
    print("=" * 100)
    print(f"  WINDOW SWEEP — Prearmed PDL | RR={RR} | Capital=₹{CASH:,.0f}")
    print(f"  Stocks: {', '.join(stock_names)}")
    print("=" * 100)

    precomputed = {}
    for period_label, days in PERIODS:
        start = TODAY - pd.Timedelta(days=days)
        precomputed[period_label] = {}
        for sym, _ in STOCKS:
            data = precompute(sym, start)
            if data and len(data) >= 3:
                precomputed[period_label][sym] = data

    results = []
    for period_label, days in PERIODS:
        for win_label, win_start, win_end in WINDOWS:
            all_trades = []
            for sym, _ in STOCKS:
                if sym not in precomputed[period_label]:
                    continue
                t = simulate(precomputed[period_label][sym], sym, win_start, win_end)
                all_trades.extend(t)
            agg = analyze(all_trades)
            results.append({
                "period": period_label,
                "window": win_label,
                **agg,
            })

    print()
    print(f"  {'Period':<6} │ {'Window':<14} │ {'Tr':>4} {'WR%':>6} {'TP':>4} {'SL':>4} │"
          f" {'P&L':>9} {'Avg/Tr':>8} {'MaxDD':>8} {'PF':>5}")
    print("  " + "─" * 87)

    for r in results:
        print(
            f"  {r['period']:<6} │ {r['window']:<14} │ {r['n']:>4} {r['wr']:>5.1f}% {r['tp']:>4} {r['sl']:>4} │"
            f" ₹{r['pnl']:>7,.0f} ₹{r['avg']:>6,.0f} ₹{r['dd']:>6,.0f} {r['pf']:>5.2f}"
        )
        if r["window"] == WINDOWS[-1][0]:
            print("  " + "─" * 87)

    best = max(results, key=lambda x: x["pnl"])
    print(f"\n  BEST: {best['period']} / {best['window']} — ₹{best['pnl']:,.0f} P&L, "
          f"{best['wr']:.1f}% WR, PF {best['pf']:.2f}")
    print("=" * 100)

    # Per-stock breakdown for the best window
    best_period = best["period"]
    best_win = next(w for w in WINDOWS if w[0] == best["window"])
    print(f"\n  PER-STOCK BREAKDOWN — {best_period} / {best['window']}")
    print(f"  {'Stock':<14} │ {'Tr':>4} {'WR%':>6} {'TP':>3} {'SL':>3} │ {'P&L':>9} {'Avg':>7}")
    print("  " + "─" * 56)
    sym_trades = {}
    for sym, _ in STOCKS:
        if sym not in precomputed[best_period]:
            continue
        t = simulate(precomputed[best_period][sym], sym, best_win[1], best_win[2])
        sym_trades[sym] = t
    sorted_syms = sorted(sym_trades.keys(), key=lambda s: sum(t["pnl"] for t in sym_trades[s]), reverse=True)
    for sym in sorted_syms:
        s = analyze(sym_trades[sym])
        print(f"  {sym:<14} │ {s['n']:>4} {s['wr']:>5.1f}% {s['tp']:>3} {s['sl']:>3} │ ₹{s['pnl']:>7,.0f} ₹{s['avg']:>5,.0f}")
    print("  " + "─" * 56)


if __name__ == "__main__":
    run()

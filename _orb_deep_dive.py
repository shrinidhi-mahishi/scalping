"""
ORB (Opening Range Breakout) Deep Dive — Nifty 50, 90 days.

Tests:
  A. Candle size:  1-min, 3-min, 5-min
  B. Range window: 15 min (09:15-09:30), 30 min (09:15-09:45), 45 min (09:15-10:00)
  C. RR sweep:     0.75, 1.0, 1.25, 1.5, 2.0
  D. Entry window: 09:30-11:00, 09:30-12:00, 09:30-13:00
  E. Volume filter: on / off
  F. Daily P&L distribution
  G. Per-stock breakdown for best combo
"""

import sys
import time as _time
from datetime import time, timedelta
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(".").resolve()))
from fetch_data import load_csv

CASH = 50_000.0
TODAY = pd.Timestamp.now().normalize()
START = TODAY - pd.Timedelta(days=90)
_F = str.maketrans({"&": ""})

RISK_PCT = 0.015
LEV_CAP = 5.0
MAX_SL_PER_DAY = 2
COOLDOWN_MIN = 30

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


def csv_1min(sym):
    return Path("data") / f"{sym.translate(_F)}_1min.csv"


def resample(df, mins):
    if mins == 1:
        return df
    return df.resample(f"{mins}min").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna(subset=["open"])


def add_indicators(df, atr_p=14, vol_p=20):
    out = df.copy()
    pc = out["close"].shift(1)
    tr = np.maximum(out["high"], pc) - np.minimum(out["low"], pc)
    out["atr"] = tr.ewm(alpha=1.0 / atr_p, min_periods=atr_p, adjust=False).mean()
    out["vol_sma"] = out["volume"].rolling(vol_p).mean()
    return out


def get_day_rows(ind):
    ind = ind[ind.index >= START]
    if len(ind) < 100:
        return []
    days = sorted(ind.index.normalize().unique())
    out = []
    for d in days:
        dr = ind[ind.index.normalize() == d]
        if len(dr) >= 10:
            out.append((d, dr))
    return out


def simulate_orb(day_list, sym, rr, range_end_time, entry_end_time, vol_filter, sl_mode="range"):
    """
    range_end_time: when the opening range ends (e.g. 09:30, 09:45, 10:00)
    entry_end_time: when to stop taking new trades
    vol_filter: require volume > vol_sma
    sl_mode: 'range' = SL = opposite side of ORB range, 'atr' = SL = 1.0×ATR
    """
    trades = []
    for td, rows in day_list:
        range_bars = rows[rows.index.map(lambda x: x.time()) < range_end_time]
        if len(range_bars) < 2:
            continue

        orb_high = float(range_bars["high"].max())
        orb_low = float(range_bars["low"].min())
        orb_range = orb_high - orb_low
        if orb_range <= 0:
            continue

        traded_dirs = set()
        pos, dsl, cd_until = None, 0, None
        cooldown = timedelta(minutes=COOLDOWN_MIN)

        active = rows[rows.index.map(lambda x: x.time()) >= range_end_time]
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
                    trades.append({**pos, "pnl": pv, "res": hit})
                    if hit == "SL":
                        dsl += 1
                    pos = None
                continue

            if t > entry_end_time or dsl >= MAX_SL_PER_DAY:
                continue
            if cd_until and ts < cd_until:
                continue

            atr = row["atr"]
            vsma = row["vol_sma"]
            if pd.isna(atr) or atr <= 0:
                continue

            if vol_filter:
                if pd.isna(vsma) or vsma <= 0 or row["volume"] <= vsma:
                    continue

            sig = None
            if "LONG" not in traded_dirs and c > orb_high:
                sig = "LONG"
            elif "SHORT" not in traded_dirs and c < orb_low:
                sig = "SHORT"

            if sig:
                if sl_mode == "range":
                    sd = max(orb_range, atr * 0.5)
                else:
                    sd = atr * 1.0

                rq = int(CASH * RISK_PCT / sd)
                mq = int(CASH * LEV_CAP / c)
                q = max(min(rq, mq), 1)
                slp = c - sd if sig == "LONG" else c + sd
                tpp = c + sd * rr if sig == "LONG" else c - sd * rr
                pos = {"sym": sym, "d": sig, "e": c, "sl": slp, "tp": tpp,
                       "atr": atr, "q": q, "date": td, "h": t.hour}
                traded_dirs.add(sig)
                cd_until = ts + cooldown
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


def run():
    print("=" * 120)
    print("  ORB DEEP DIVE — Nifty 50 × 90 days")
    print(f"  Period: {START.date()} → {TODAY.date()}")
    print("=" * 120)

    # Load 1-min data
    print("\n[1] Loading 1-min data ...\n")
    raw_data = {}
    for sym, sec in NIFTY50:
        p = csv_1min(sym)
        if not p.exists():
            continue
        try:
            df = load_csv(str(p))
            if len(df) > 200:
                raw_data[sym] = {"df": df, "sector": sec}
        except Exception:
            pass
    print(f"  {len(raw_data)} stocks loaded\n")

    # ─── PART A: Candle size comparison (1m, 3m, 5m) ─────────────────────
    print("=" * 120)
    print("  PART A — CANDLE SIZE: 1-min vs 3-min vs 5-min (ORB 15min range, RR=1.0)")
    print("=" * 120)

    for mins in [1, 3, 5]:
        atr_p = {1: 42, 3: 14, 5: 9}[mins]
        vol_p = {1: 60, 3: 20, 5: 12}[mins]
        all_trades = []
        for sym, info in raw_data.items():
            df = resample(info["df"], mins)
            ind = add_indicators(df, atr_p, vol_p)
            dl = get_day_rows(ind)
            t = simulate_orb(dl, sym, rr=1.0,
                             range_end_time=time(9, 30),
                             entry_end_time=time(13, 0),
                             vol_filter=True)
            all_trades.extend(t)
        a = analyze(all_trades)
        print(f"  {mins}-min:  {a['n']:>5} trades  WR={a['wr']:>5.1f}%  "
              f"P&L=₹{a['pnl']:>9,.0f}  DD=₹{a['dd']:>8,.0f}  PF={a['pf']:.2f}  Avg=₹{a['avg']:>5,.0f}")

    # ─── PART B: Range window sweep ──────────────────────────────────────
    print(f"\n{'=' * 120}")
    print("  PART B — RANGE WINDOW: 15min vs 30min vs 45min (1-min candles, RR=1.0)")
    print("=" * 120)

    range_configs = [
        ("15 min (09:15-09:30)", time(9, 30)),
        ("30 min (09:15-09:45)", time(9, 45)),
        ("45 min (09:15-10:00)", time(10, 0)),
    ]

    for label, rend in range_configs:
        all_trades = []
        for sym, info in raw_data.items():
            ind = add_indicators(info["df"], 42, 60)
            dl = get_day_rows(ind)
            t = simulate_orb(dl, sym, rr=1.0, range_end_time=rend,
                             entry_end_time=time(13, 0), vol_filter=True)
            all_trades.extend(t)
        a = analyze(all_trades)
        print(f"  {label:30s}  {a['n']:>5} trades  WR={a['wr']:>5.1f}%  "
              f"P&L=₹{a['pnl']:>9,.0f}  DD=₹{a['dd']:>8,.0f}  PF={a['pf']:.2f}")

    # ─── PART C: Full sweep — candle × range × RR × volume × entry end ──
    print(f"\n{'=' * 120}")
    print("  PART C — FULL PARAMETER SWEEP (top 20 combos by P&L)")
    print("=" * 120)

    candle_opts = [1, 3]
    range_opts = [time(9, 30), time(9, 45)]
    rr_opts = [0.75, 1.0, 1.25, 1.5, 2.0]
    entry_opts = [time(11, 0), time(12, 0), time(13, 0)]
    vol_opts = [True, False]

    precomputed = {}
    for mins in candle_opts:
        atr_p = {1: 42, 3: 14}[mins]
        vol_p = {1: 60, 3: 20}[mins]
        precomputed[mins] = {}
        for sym, info in raw_data.items():
            df = resample(info["df"], mins)
            ind = add_indicators(df, atr_p, vol_p)
            dl = get_day_rows(ind)
            if dl:
                precomputed[mins][sym] = dl

    sweep = []
    total = len(candle_opts) * len(range_opts) * len(rr_opts) * len(entry_opts) * len(vol_opts)
    done = 0

    for mins, rend, rr, entry_e, vf in product(candle_opts, range_opts, rr_opts, entry_opts, vol_opts):
        all_trades = []
        stock_pnl = {}
        for sym in precomputed[mins]:
            dl = precomputed[mins][sym]
            t = simulate_orb(dl, sym, rr, rend, entry_e, vf)
            all_trades.extend(t)
            stock_pnl[sym] = sum(x["pnl"] for x in t)

        a = analyze(all_trades)
        rend_m = (rend.hour - 9) * 60 + rend.minute - 15
        sweep.append({
            "candle": f"{mins}m", "range": f"{rend_m}min",
            "rr": rr, "entry_end": f"{entry_e:%H:%M}",
            "vol_filter": "Yes" if vf else "No",
            **a, "stock_pnl": stock_pnl,
        })
        done += 1
        if done % 20 == 0:
            print(f"  ... {done}/{total} combos tested", flush=True)

    print(f"  ... {total}/{total} combos tested\n")

    sweep.sort(key=lambda x: x["pnl"], reverse=True)

    print(f"  {'#':>3} {'Candle':>6} {'Range':>6} {'RR':>5} {'EntryEnd':>9} {'Vol':>4} │ "
          f"{'Trades':>6} {'WR%':>6} {'P&L':>10} {'DD':>10} {'PF':>5} {'Avg':>7}")
    print("  " + "─" * 100)

    for i, r in enumerate(sweep[:20], 1):
        print(f"  {i:>3} {r['candle']:>6} {r['range']:>6} {r['rr']:>5.2f} "
              f"{r['entry_end']:>9} {r['vol_filter']:>4} │ "
              f"{r['n']:>6} {r['wr']:>5.1f}% ₹{r['pnl']:>8,.0f} ₹{r['dd']:>8,.0f} "
              f"{r['pf']:>5.2f} ₹{r['avg']:>5,.0f}")

    # ─── PART D: Best combo — detailed analysis ─────────────────────────
    best = sweep[0]
    print(f"\n{'=' * 120}")
    print(f"  BEST COMBO: {best['candle']} candle, {best['range']} range, "
          f"RR={best['rr']}, entry until {best['entry_end']}, vol_filter={best['vol_filter']}")
    print("=" * 120)
    print(f"  Trades: {best['n']}  WR: {best['wr']:.1f}%  P&L: ₹{best['pnl']:,.0f}")
    print(f"  TP: {best['tp']}  SL: {best['sl']}  DD: ₹{best['dd']:,.0f}  PF: {best['pf']:.2f}")

    # Top 10 stocks
    sp = best["stock_pnl"]
    ranked = sorted(sp.items(), key=lambda x: x[1], reverse=True)
    top10_pnl = sum(p for _, p in ranked[:10])

    print(f"\n  Top 10 stocks (combined P&L: ₹{top10_pnl:,.0f}):")
    print(f"  {'#':>3} {'Stock':<14} {'Sector':<12} {'P&L':>10}")
    print("  " + "─" * 45)
    for i, (sym, pnl) in enumerate(ranked[:10], 1):
        sec = raw_data.get(sym, {}).get("sector", "?")
        print(f"  {i:>3} {sym:<14} {sec:<12} ₹{pnl:>8,.0f} ★")

    print(f"\n  Bottom 5:")
    for sym, pnl in ranked[-5:]:
        sec = raw_data.get(sym, {}).get("sector", "?")
        print(f"      {sym:<14} {sec:<12} ₹{pnl:>8,.0f}")

    # ─── PART E: Re-run best combo and analyze daily P&L ─────────────────
    best_mins = int(best["candle"].replace("m", ""))
    best_rend = time(9, 30) if best["range"] == "15min" else time(9, 45)
    best_entry_e = time(int(best["entry_end"].split(":")[0]), int(best["entry_end"].split(":")[1]))
    best_vf = best["vol_filter"] == "Yes"

    all_trades_best = []
    for sym in precomputed[best_mins]:
        dl = precomputed[best_mins][sym]
        t = simulate_orb(dl, sym, best["rr"], best_rend, best_entry_e, best_vf)
        all_trades_best.extend(t)

    if all_trades_best:
        tdf = pd.DataFrame(all_trades_best)
        daily = tdf.groupby("date")["pnl"].sum()

        print(f"\n  DAILY P&L DISTRIBUTION:")
        print(f"  Positive days: {int((daily > 0).sum())} / {len(daily)}")
        print(f"  Negative days: {int((daily < 0).sum())} / {len(daily)}")
        print(f"  Avg daily P&L: ₹{daily.mean():,.0f}")
        print(f"  Median daily:  ₹{daily.median():,.0f}")
        print(f"  Best day:      ₹{daily.max():,.0f}")
        print(f"  Worst day:     ₹{daily.min():,.0f}")
        print(f"  Std dev:       ₹{daily.std():,.0f}")

        # Win/loss streaks
        signs = (daily > 0).astype(int)
        streaks = signs.groupby((signs != signs.shift()).cumsum())
        win_streaks = [len(g) for _, g in streaks if g.iloc[0] == 1]
        loss_streaks = [len(g) for _, g in streaks if g.iloc[0] == 0]
        print(f"  Max win streak:  {max(win_streaks) if win_streaks else 0} days")
        print(f"  Max loss streak: {max(loss_streaks) if loss_streaks else 0} days")

        # Hour distribution
        hr = tdf.groupby("h")["pnl"].agg(["count", "sum"]).rename(
            columns={"count": "cnt", "sum": "total"})
        print(f"\n  ENTRY HOUR DISTRIBUTION:")
        for h, row in hr.iterrows():
            bar = "█" * max(1, int(row["cnt"] / 3))
            print(f"    {h:02d}:xx  {int(row['cnt']):>4} trades  ₹{row['total']:>8,.0f}  {bar}")

    # ─── PART F: Compare with current 3-min PDL ─────────────────────────
    print(f"\n{'=' * 120}")
    print("  FINAL COMPARISON: Best ORB vs Current 3-min PDL")
    print("=" * 120)
    print(f"  {'Metric':<25} {'3-min PDL (current)':>20} {'ORB (best combo)':>20}")
    print("  " + "─" * 70)
    pdl_vals = {"n": 218, "wr": 40.8, "pnl": 59122, "dd": -4902, "pf": 1.88, "avg": 271}
    orb_vals = best
    for key, label in [("n", "Trades"), ("wr", "Win Rate %"), ("pnl", "P&L ₹"),
                        ("dd", "Max Drawdown ₹"), ("pf", "Profit Factor"), ("avg", "Avg/Trade ₹")]:
        pv = pdl_vals[key]
        ov = orb_vals[key]
        if key in ("pnl", "dd", "avg"):
            print(f"  {label:<25} ₹{pv:>17,.0f} ₹{ov:>17,.0f}")
        elif key == "wr":
            print(f"  {label:<25} {pv:>19.1f}% {ov:>19.1f}%")
        elif key == "pf":
            print(f"  {label:<25} {pv:>20.2f} {ov:>20.2f}")
        else:
            print(f"  {label:<25} {pv:>20} {ov:>20}")

    print(f"\n  Top-10 P&L:  PDL=₹59,122  |  ORB=₹{top10_pnl:,.0f}")
    diff = top10_pnl - 59122
    print(f"  Difference: ₹{diff:+,.0f}")
    if diff > 0:
        print(f"  → ORB beats PDL on top-10 P&L")
    else:
        print(f"  → PDL still wins on top-10 P&L")

    if best["dd"] < pdl_vals["dd"] * 2:
        print(f"  ⚠ ORB drawdown (₹{best['dd']:,.0f}) is manageable")
    else:
        print(f"  ⚠ ORB drawdown (₹{best['dd']:,.0f}) is SIGNIFICANTLY worse than PDL (₹{pdl_vals['dd']:,.0f})")

    print(f"\n  STOCKS config (ORB top 10):")
    print("  STOCKS = [")
    for sym, pnl in ranked[:10]:
        sec = raw_data.get(sym, {}).get("sector", "?")
        print(f'      ("{sym}", "{sec}"),')
    print("  ]")
    print("=" * 120)


if __name__ == "__main__":
    run()

"""1-min vs 3-min PDL Strategy Comparison — 90-day backtest on top 10 stocks."""

import sys
import time as _time
from datetime import time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(".").resolve()))
from fetch_data import AngelOneClient, load_csv, save_csv
from live_signals import (
    STOCKS, RR, RISK_PCT, LEV_CAP,
    PDL_SL_MULT, PDL_LONG_RSI_MIN, PDL_SHORT_RSI_MAX,
    MAX_SL_PER_DAY, ENTRY_AM, ENTRY_PM, COOLDOWN,
    compute_indicators,
)

CASH = 50_000.0
TODAY = pd.Timestamp.now().normalize()
START = TODAY - pd.Timedelta(days=90)
_F = str.maketrans({"&": ""})


def csv_path_1min(sym):
    fname = sym.translate(_F)
    return Path("data") / f"{fname}_1min.csv"


def fetch_1min_data():
    client = AngelOneClient()
    client.connect()

    for sym, _ in STOCKS:
        p = csv_path_1min(sym)
        if p.exists():
            df = load_csv(str(p))
            if df.index.max().date() >= (TODAY - timedelta(days=2)).date():
                print(f"  {sym:14s} — cached ({len(df)} bars)")
                continue

        print(f"  {sym:14s} — fetching 90d 1-min candles ...", end="", flush=True)
        try:
            df = client.fetch_history(sym, days=100, interval="ONE_MINUTE")
            Path("data").mkdir(exist_ok=True)
            save_csv(df, str(p))
        except Exception as e:
            print(f"  FAILED: {e}")
        _time.sleep(1)


def resample_to_3min(df_1min: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-min OHLCV to 3-min bars."""
    return df_1min.resample("3min").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna(subset=["open"])


def precompute(sym, df: pd.DataFrame):
    try:
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
    except Exception:
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
        pdl_dirs, cd_until, pos, pc, dsl = set(), None, None, None, 0

        for i in range(len(today_rows)):
            row = today_rows.iloc[i]
            ts = today_rows.index[i]
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
                    trades.append({**pos, "pnl": pv, "res": hit, "exit_t": t})
                    if hit == "SL":
                        dsl += 1
                    pos, pc = None, c
                    continue
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
                tpp = c + sd * RR if sig == "LONG" else c - sd * RR
                pos = {
                    "sym": sym, "d": sig, "trig": "PDL", "e": c, "sl": slp,
                    "tp": tpp, "atr": atr, "q": q, "h": t.hour, "min": t.minute,
                    "date": today, "entry_t": t,
                }
                cd_until = ts + COOLDOWN
                pdl_dirs.add(sig)
            pc = c
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
    print("=" * 100)
    print("  1-MIN vs 3-MIN CANDLE COMPARISON — PDL Strategy")
    print(f"  Period: {START.date()} to {TODAY.date()} (90 days)")
    print(f"  RR={RR}, SL={PDL_SL_MULT}×ATR, CD={int(COOLDOWN.total_seconds()//60)}min")
    print(f"  Entry: {ENTRY_AM[0]:%H:%M}-{ENTRY_AM[1]:%H:%M}")
    print("=" * 100)

    print("\n[1] Fetching 1-min data from Angel One API ...")
    fetch_1min_data()

    trades_1m_all, trades_3m_all = [], []
    res_1m, res_3m = [], []

    print("\n[2] Running backtests (resampling 1-min → 3-min for fair comparison) ...")
    for sym, _ in STOCKS:
        p = csv_path_1min(sym)
        if not p.exists():
            print(f"  {sym:14s}  SKIP — no 1-min data")
            continue
        df_1m = load_csv(str(p))
        df_3m = resample_to_3min(df_1m)

        dd1 = precompute(sym, df_1m)
        dd3 = precompute(sym, df_3m)

        a1_r = {"n": 0, "pnl": 0.0, "wr": 0.0}
        a3_r = {"n": 0, "pnl": 0.0, "wr": 0.0}

        if dd1:
            t1 = simulate(dd1, sym)
            trades_1m_all.extend(t1)
            a1_r = analyze(t1)
            res_1m.append({"sym": sym, **a1_r})
        if dd3:
            t3 = simulate(dd3, sym)
            trades_3m_all.extend(t3)
            a3_r = analyze(t3)
            res_3m.append({"sym": sym, **a3_r})

        print(f"  {sym:14s}  1min: {a1_r['n']:>3} trades ₹{a1_r['pnl']:>8,.0f}  "
              f"3min: {a3_r['n']:>3} trades ₹{a3_r['pnl']:>8,.0f}")

    a1 = analyze(trades_1m_all)
    a3 = analyze(trades_3m_all)

    print("\n" + "=" * 100)
    print("  HEAD-TO-HEAD COMPARISON")
    print("=" * 100)
    print(f"  {'Metric':<20} {'3-MIN (current)':>20}  {'1-MIN (proposed)':>20}  {'Diff':>15}")
    print("  " + "─" * 80)

    for key, label, fmt in [
        ("n",    "Trades",          "d"),
        ("wr",   "Win Rate %",      ".1f"),
        ("pnl",  "Total P&L ₹",     ",.0f"),
        ("tp",   "TP Hits",         "d"),
        ("sl",   "SL Hits",         "d"),
        ("dd",   "Max Drawdown ₹",  ",.0f"),
        ("pf",   "Profit Factor",   ".2f"),
        ("avg",  "Avg/Trade ₹",     ",.0f"),
    ]:
        v3, v1 = a3[key], a1[key]
        diff = v1 - v3
        sign = "+" if diff >= 0 else ""
        print(f"  {label:<20} {format(v3, fmt):>20s}  {format(v1, fmt):>20s}  {sign}{format(diff, fmt):>12s}")

    print("\n" + "=" * 100)
    print("  PER-STOCK COMPARISON")
    print("=" * 100)
    print(f"  {'Stock':<14} │ {'3m P&L':>10} {'#':>4} {'WR%':>5} │ {'1m P&L':>10} {'#':>4} {'WR%':>5} │ {'Δ P&L':>10}")
    print("  " + "─" * 80)

    s3 = {r["sym"]: r for r in res_3m}
    s1 = {r["sym"]: r for r in res_1m}

    for sym, _ in STOCKS:
        r3 = s3.get(sym, {"pnl": 0, "n": 0, "wr": 0})
        r1 = s1.get(sym, {"pnl": 0, "n": 0, "wr": 0})
        dp = r1["pnl"] - r3["pnl"]
        sign = "+" if dp >= 0 else ""
        print(f"  {sym:<14} │ ₹{r3['pnl']:>8,.0f} {r3['n']:>4} {r3['wr']:>4.1f}% "
              f"│ ₹{r1['pnl']:>8,.0f} {r1['n']:>4} {r1['wr']:>4.1f}% "
              f"│ {sign}₹{abs(dp):>7,.0f}")

    if trades_1m_all and trades_3m_all:
        print("\n" + "=" * 100)
        print("  ENTRY TIMING — WHEN DO SIGNALS FIRE?")
        print("=" * 100)

        df1 = pd.DataFrame(trades_1m_all)
        df3 = pd.DataFrame(trades_3m_all)

        for label, df in [("3-MIN", df3), ("1-MIN", df1)]:
            hr = df.groupby("h")["pnl"].agg(["count", "sum"]).rename(
                columns={"count": "cnt", "sum": "total"})
            print(f"\n  {label}:")
            for h, row in hr.iterrows():
                bar = "█" * int(row["cnt"])
                print(f"    {h:02d}:xx  {int(row['cnt']):>3} trades  ₹{row['total']:>8,.0f}  {bar}")

        # Same-day signal timing comparison
        df1["key"] = df1["sym"] + "_" + df1["date"].astype(str) + "_" + df1["d"]
        df3["key"] = df3["sym"] + "_" + df3["date"].astype(str) + "_" + df3["d"]

        common = set(df1["key"]) & set(df3["key"])
        if common:
            timing = []
            for k in common:
                r1 = df1[df1["key"] == k].iloc[0]
                r3 = df3[df3["key"] == k].iloc[0]
                t1 = r1["h"] * 60 + r1["min"]
                t3 = r3["h"] * 60 + r3["min"]
                timing.append({"sym": r1["sym"], "dir": r1["d"],
                               "1m_time": f"{r1['h']:02d}:{r1['min']:02d}",
                               "3m_time": f"{r3['h']:02d}:{r3.get('min', 0):02d}",
                               "delta_min": t3 - t1,
                               "1m_entry": r1["e"], "3m_entry": r3["e"],
                               "price_diff": r1["e"] - r3["e"],
                               "1m_res": r1["res"], "3m_res": r3["res"]})

            tdf = pd.DataFrame(timing)
            avg_delta = tdf["delta_min"].mean()
            earlier = int((tdf["delta_min"] > 0).sum())
            same = int((tdf["delta_min"] == 0).sum())
            later = int((tdf["delta_min"] < 0).sum())

            print(f"\n  Matched signals (same stock + day + direction): {len(common)}")
            print(f"  1-min fires earlier: {earlier}  |  same time: {same}  |  later: {later}")
            print(f"  Average time advantage: {avg_delta:.1f} minutes earlier on 1-min")

            if len(tdf) <= 30:
                print(f"\n  {'Sym':<14} {'Dir':<6} {'1m Time':>8} {'3m Time':>8} {'Δmin':>5} "
                      f"{'1m Entry':>10} {'3m Entry':>10} {'1m':>4} {'3m':>4}")
                print("  " + "─" * 80)
                for _, r in tdf.sort_values("delta_min", ascending=False).iterrows():
                    print(f"  {r['sym']:<14} {r['dir']:<6} {r['1m_time']:>8} {r['3m_time']:>8} "
                          f"{r['delta_min']:>+4.0f}m "
                          f"₹{r['1m_entry']:>8,.1f} ₹{r['3m_entry']:>8,.1f} "
                          f"{r['1m_res']:>4} {r['3m_res']:>4}")

    # False breakout analysis: trades that only appear on 1-min
    only_1m = set(df1["key"]) - set(df3["key"]) if trades_1m_all and trades_3m_all else set()
    if only_1m:
        extra = df1[df1["key"].isin(only_1m)]
        extra_wins = int((extra["pnl"] > 0).sum())
        extra_pnl = extra["pnl"].sum()
        print(f"\n  1-MIN-ONLY signals (not in 3-min): {len(only_1m)}")
        print(f"    Win rate: {extra_wins}/{len(only_1m)} = {extra_wins/len(only_1m)*100:.1f}%")
        print(f"    P&L: ₹{extra_pnl:,.0f}")
        if extra_pnl < 0:
            print("    → These extra signals LOSE money — confirms 1-min noise")
        else:
            print("    → These extra signals are profitable — 1-min catches real moves")

    print("\n" + "=" * 100)
    winner = "1-MIN" if a1["pnl"] > a3["pnl"] else "3-MIN"
    print(f"  VERDICT: {winner} is better")
    if winner == "3-MIN":
        print("  1-min adds noise/false breakouts that erode P&L. Stick with 3-min candles.")
    else:
        pct = ((a1["pnl"] - a3["pnl"]) / abs(a3["pnl"]) * 100) if a3["pnl"] != 0 else 0
        print(f"  1-min improves P&L by ₹{a1['pnl'] - a3['pnl']:,.0f} ({pct:+.0f}%).")
        if a1["dd"] < a3["dd"] * 1.3:
            print("  Drawdown is acceptable. Consider switching.")
        else:
            print(f"  BUT drawdown worsens significantly (₹{a1['dd']:,.0f} vs ₹{a3['dd']:,.0f}). Risky.")
    print("=" * 100)


if __name__ == "__main__":
    run()

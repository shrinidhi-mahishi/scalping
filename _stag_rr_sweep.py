"""
Sweep STAG and RR parameters to find optimal combination.
Tests on new top 10 stocks over 90 days with full hybrid filters.
"""

import sys
from datetime import timedelta, time
from pathlib import Path
from itertools import product
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(".").resolve()))
from fetch_data import load_csv
from live_signals import (
    STOCKS, RISK_PCT, LEV_CAP,
    PDL_SL_MULT, MOM_SL_MULT, MOM_VOL_MULT, MOM_BODY_RATIO,
    MOM_LONG_RSI, MOM_SHORT_RSI, PDL_LONG_RSI_MIN, PDL_SHORT_RSI_MAX,
    MAX_SL_PER_DAY, ENTRY_AM, ENTRY_PM, compute_indicators,
)

CASH = 50_000.0
_F = str.maketrans({"&": ""})

RR_VALUES = [1.25, 1.5, 1.75, 2.0, 2.5]
STAG_BARS_VALUES = [6, 8, 10, 12, 15, 999]  # 999 = no stag exit
STAG_ATR_VALUES = [0.1, 0.2, 0.3, 0.5]
COOLDOWN_VALUES = [15, 30, 45, 60]


def csv_path(sym):
    fname = sym.translate(_F)
    p = Path("data") / f"{fname}_3min.csv"
    return p if p.exists() else Path("data") / f"{fname.lower()}_3min.csv"


def in_entry_window(t):
    if ENTRY_AM and ENTRY_AM[0] <= t <= ENTRY_AM[1]:
        return True
    if ENTRY_PM and ENTRY_PM[0] <= t <= ENTRY_PM[1]:
        return True
    return False


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


def safe_vr(row):
    v = row.get("vol_ratio", 0)
    return 0 if pd.isna(v) else v


def safe_br(row):
    v = row.get("body_ratio", 0)
    return 0 if pd.isna(v) else v


def simulate(day_data, sym, rr, stag_bars, stag_atr, cooldown_min):
    trades = []
    cooldown = timedelta(minutes=cooldown_min)
    for today, pdl_h, pdl_l, today_rows, prev_rows in day_data:
        pdl_dirs, cd_until, pos, bh, pc, dsl = set(), None, None, 0, None, 0
        prev_vol_r = None

        all_rows = pd.concat([prev_rows, today_rows]) if prev_rows is not None else today_rows
        atr_20_series = all_rows["atr"].rolling(20).mean()

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
                    prev_vol_r = safe_vr(row)
                    continue
                if stag_bars < 999 and bh >= stag_bars:
                    u = (c - pos["e"]) if pos["d"] == "LONG" else (pos["e"] - c)
                    if u < stag_atr * pos["atr"]:
                        trades.append({**pos, "pnl": u * pos["q"], "res": "STAG"})
                        pos, bh = None, 0
                pc = c
                prev_vol_r = safe_vr(row)
                continue

            if not in_entry_window(t) or (cd_until and ts < cd_until) or dsl >= MAX_SL_PER_DAY:
                pc = c
                prev_vol_r = safe_vr(row)
                continue

            rsi, atr, vwap, vsma = row["rsi"], row["atr"], row["vwap"], row["vol_sma"]
            if pd.isna(rsi) or pd.isna(atr) or atr <= 0 or pd.isna(vwap) or pd.isna(vsma):
                pc = c
                prev_vol_r = safe_vr(row)
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
                br = safe_br(row)
                vr = safe_vr(row)
                atr_idx = all_rows.index.get_loc(ts) if ts in all_rows.index else -1
                atr_20 = atr_20_series.iloc[atr_idx] if atr_idx >= 0 and not pd.isna(atr_20_series.iloc[atr_idx]) else atr
                atr_ratio = atr / atr_20 if atr_20 > 0 else 1.0
                body_threshold = 0.72 if atr_ratio > 1.1 else MOM_BODY_RATIO

                if (prev_vol_r is not None
                        and prev_vol_r >= MOM_VOL_MULT * 0.8
                        and vr >= MOM_VOL_MULT * 0.8
                        and br >= body_threshold):
                    if (c > row["open"] and c > vwap
                            and MOM_LONG_RSI[0] <= rsi <= MOM_LONG_RSI[1]):
                        sig, trig = "LONG", "MOM"
                    elif (c < row["open"] and c < vwap
                          and MOM_SHORT_RSI[0] <= rsi <= MOM_SHORT_RSI[1]):
                        sig, trig = "SHORT", "MOM"

            if sig:
                sm = PDL_SL_MULT if trig == "PDL" else MOM_SL_MULT
                atr_idx = all_rows.index.get_loc(ts) if ts in all_rows.index else -1
                atr_20 = atr_20_series.iloc[atr_idx] if atr_idx >= 0 and not pd.isna(atr_20_series.iloc[atr_idx]) else atr
                atr_ratio = atr / atr_20 if atr_20 > 0 else 1.0
                if atr_ratio > 1.3:
                    sm *= 1.2
                elif atr_ratio < 0.7:
                    sm *= 0.9

                sd = atr * sm
                rq = int(CASH * RISK_PCT / sd)
                mq = int(CASH * LEV_CAP / c)
                q = max(min(rq, mq), 1)
                slp = c - sd if sig == "LONG" else c + sd
                tpp = c + sd * rr if sig == "LONG" else c - sd * rr
                pos = {
                    "sym": sym, "d": sig, "trig": trig, "e": c, "sl": slp,
                    "tp": tpp, "atr": atr, "q": q, "h": t.hour, "date": today,
                }
                bh = 0
                cd_until = ts + cooldown
                if trig == "PDL":
                    pdl_dirs.add(sig)
            pc = c
            prev_vol_r = safe_vr(row)
    return trades


def run():
    stock_data = {}
    for sym, _ in STOCKS:
        dd = precompute(sym)
        if dd:
            stock_data[sym] = dd
    print(f"  {len(stock_data)}/{len(STOCKS)} stocks loaded\n")

    # ── Phase 1: RR sweep (keep other params at current) ──
    print("=" * 100)
    print("  PHASE 1: RR SWEEP (stag=10 bars, 0.2×ATR, cooldown=45min)")
    print("=" * 100)
    print(f"  {'RR':>5} {'Trades':>7} {'WR':>6} {'P&L':>11} {'MaxDD':>10} {'PF':>6} {'TP':>4} {'SL':>4} {'Stag':>4} {'Avg':>8}")
    print("  " + "─" * 90)

    for rr in RR_VALUES:
        all_trades = []
        for sym, dd in stock_data.items():
            all_trades.extend(simulate(dd, sym, rr, 10, 0.2, 45))
        if not all_trades:
            continue
        df = pd.DataFrame(all_trades)
        t = len(df)
        tp = len(df[df["res"] == "TP"])
        sl = len(df[df["res"] == "SL"])
        stag = len(df[df["res"] == "STAG"])
        pnl = df["pnl"].sum()
        wr = tp / t * 100
        maxdd = (df["pnl"].cumsum() - df["pnl"].cumsum().cummax()).min()
        gw = df[df["pnl"] > 0]["pnl"].sum()
        gl = abs(df[df["pnl"] < 0]["pnl"].sum())
        pf = gw / gl if gl > 0 else 9.99
        avg = pnl / t
        marker = " ◄" if rr == 1.75 else ""
        print(f"  {rr:>5.2f} {t:>7} {wr:>5.1f}% ₹{pnl:>10,.0f} ₹{maxdd:>9,.0f} {pf:>5.2f} {tp:>4} {sl:>4} {stag:>4} ₹{avg:>6,.0f}{marker}")

    # ── Phase 2: STAG bars sweep ──
    print(f"\n{'=' * 100}")
    print("  PHASE 2: STAG BARS SWEEP (RR=1.75, stag_atr=0.2, cooldown=45min)")
    print("=" * 100)
    print(f"  {'Bars':>5} {'Trades':>7} {'WR':>6} {'P&L':>11} {'MaxDD':>10} {'PF':>6} {'TP':>4} {'SL':>4} {'Stag':>4} {'Avg':>8}")
    print("  " + "─" * 90)

    for sb in STAG_BARS_VALUES:
        all_trades = []
        for sym, dd in stock_data.items():
            all_trades.extend(simulate(dd, sym, 1.75, sb, 0.2, 45))
        if not all_trades:
            continue
        df = pd.DataFrame(all_trades)
        t = len(df)
        tp = len(df[df["res"] == "TP"])
        sl = len(df[df["res"] == "SL"])
        stag = len(df[df["res"] == "STAG"])
        pnl = df["pnl"].sum()
        wr = tp / t * 100
        maxdd = (df["pnl"].cumsum() - df["pnl"].cumsum().cummax()).min()
        gw = df[df["pnl"] > 0]["pnl"].sum()
        gl = abs(df[df["pnl"] < 0]["pnl"].sum())
        pf = gw / gl if gl > 0 else 9.99
        avg = pnl / t
        lbl = "none" if sb == 999 else str(sb)
        marker = " ◄" if sb == 10 else ""
        print(f"  {lbl:>5} {t:>7} {wr:>5.1f}% ₹{pnl:>10,.0f} ₹{maxdd:>9,.0f} {pf:>5.2f} {tp:>4} {sl:>4} {stag:>4} ₹{avg:>6,.0f}{marker}")

    # ── Phase 3: STAG ATR threshold sweep ──
    print(f"\n{'=' * 100}")
    print("  PHASE 3: STAG ATR THRESHOLD SWEEP (RR=1.75, stag_bars=10, cooldown=45min)")
    print("=" * 100)
    print(f"  {'ATR':>5} {'Trades':>7} {'WR':>6} {'P&L':>11} {'MaxDD':>10} {'PF':>6} {'TP':>4} {'SL':>4} {'Stag':>4} {'Avg':>8}")
    print("  " + "─" * 90)

    for sa in STAG_ATR_VALUES:
        all_trades = []
        for sym, dd in stock_data.items():
            all_trades.extend(simulate(dd, sym, 1.75, 10, sa, 45))
        if not all_trades:
            continue
        df = pd.DataFrame(all_trades)
        t = len(df)
        tp = len(df[df["res"] == "TP"])
        sl = len(df[df["res"] == "SL"])
        stag = len(df[df["res"] == "STAG"])
        pnl = df["pnl"].sum()
        wr = tp / t * 100
        maxdd = (df["pnl"].cumsum() - df["pnl"].cumsum().cummax()).min()
        gw = df[df["pnl"] > 0]["pnl"].sum()
        gl = abs(df[df["pnl"] < 0]["pnl"].sum())
        pf = gw / gl if gl > 0 else 9.99
        avg = pnl / t
        marker = " ◄" if sa == 0.2 else ""
        print(f"  {sa:>5.1f} {t:>7} {wr:>5.1f}% ₹{pnl:>10,.0f} ₹{maxdd:>9,.0f} {pf:>5.2f} {tp:>4} {sl:>4} {stag:>4} ₹{avg:>6,.0f}{marker}")

    # ── Phase 4: Cooldown sweep ──
    print(f"\n{'=' * 100}")
    print("  PHASE 4: COOLDOWN SWEEP (RR=1.75, stag=10 bars, 0.2×ATR)")
    print("=" * 100)
    print(f"  {'CD':>5} {'Trades':>7} {'WR':>6} {'P&L':>11} {'MaxDD':>10} {'PF':>6} {'TP':>4} {'SL':>4} {'Stag':>4} {'Avg':>8}")
    print("  " + "─" * 90)

    for cd in COOLDOWN_VALUES:
        all_trades = []
        for sym, dd in stock_data.items():
            all_trades.extend(simulate(dd, sym, 1.75, 10, 0.2, cd))
        if not all_trades:
            continue
        df = pd.DataFrame(all_trades)
        t = len(df)
        tp = len(df[df["res"] == "TP"])
        sl = len(df[df["res"] == "SL"])
        stag = len(df[df["res"] == "STAG"])
        pnl = df["pnl"].sum()
        wr = tp / t * 100
        maxdd = (df["pnl"].cumsum() - df["pnl"].cumsum().cummax()).min()
        gw = df[df["pnl"] > 0]["pnl"].sum()
        gl = abs(df[df["pnl"] < 0]["pnl"].sum())
        pf = gw / gl if gl > 0 else 9.99
        avg = pnl / t
        marker = " ◄" if cd == 45 else ""
        print(f"  {cd:>4}m {t:>7} {wr:>5.1f}% ₹{pnl:>10,.0f} ₹{maxdd:>9,.0f} {pf:>5.2f} {tp:>4} {sl:>4} {stag:>4} ₹{avg:>6,.0f}{marker}")

    # ── Phase 5: Best combo search ──
    print(f"\n{'=' * 100}")
    print("  PHASE 5: TOP 10 COMBINATIONS (full grid search)")
    print("=" * 100)

    results = []
    for rr, sb, sa, cd in product(RR_VALUES, STAG_BARS_VALUES, STAG_ATR_VALUES, COOLDOWN_VALUES):
        all_trades = []
        for sym, dd in stock_data.items():
            all_trades.extend(simulate(dd, sym, rr, sb, sa, cd))
        if not all_trades:
            continue
        df = pd.DataFrame(all_trades)
        t = len(df)
        tp = len(df[df["res"] == "TP"])
        sl = len(df[df["res"] == "SL"])
        stag = len(df[df["res"] == "STAG"])
        pnl = df["pnl"].sum()
        wr = tp / t * 100
        maxdd = (df["pnl"].cumsum() - df["pnl"].cumsum().cummax()).min()
        gw = df[df["pnl"] > 0]["pnl"].sum()
        gl = abs(df[df["pnl"] < 0]["pnl"].sum())
        pf = gw / gl if gl > 0 else 9.99
        avg = pnl / t
        results.append({
            "rr": rr, "stag_bars": sb, "stag_atr": sa, "cd": cd,
            "trades": t, "wr": wr, "pnl": pnl, "maxdd": maxdd,
            "pf": pf, "tp": tp, "sl": sl, "stag": stag, "avg": avg,
        })

    results.sort(key=lambda x: x["pnl"], reverse=True)

    print(f"  {'#':>3} {'RR':>5} {'StgB':>5} {'StgA':>5} {'CD':>4} {'Trades':>7} {'WR':>6} "
          f"{'P&L':>11} {'MaxDD':>10} {'PF':>6} {'TP':>4} {'SL':>4} {'Stag':>4} {'Avg':>8}")
    print("  " + "─" * 100)
    for i, r in enumerate(results[:10]):
        sb_lbl = "none" if r["stag_bars"] == 999 else str(r["stag_bars"])
        print(f"  {i+1:>3} {r['rr']:>5.2f} {sb_lbl:>5} {r['stag_atr']:>5.1f} {r['cd']:>3}m "
              f"{r['trades']:>7} {r['wr']:>5.1f}% ₹{r['pnl']:>10,.0f} ₹{r['maxdd']:>9,.0f} "
              f"{r['pf']:>5.2f} {r['tp']:>4} {r['sl']:>4} {r['stag']:>4} ₹{r['avg']:>6,.0f}")

    # Show current config position
    current = [r for r in results if r["rr"] == 1.75 and r["stag_bars"] == 10
               and r["stag_atr"] == 0.2 and r["cd"] == 45]
    if current:
        rank = results.index(current[0]) + 1
        print(f"\n  Current config (RR=1.75, stag=10/0.2, CD=45m) ranks #{rank}/{len(results)}")

    print("=" * 100)


if __name__ == "__main__":
    run()

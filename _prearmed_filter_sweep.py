"""
Lightweight filter sweep for Prearmed PDL.

Goal:
  Improve fake-breakout quality without reintroducing lag.

Filters tested:
  - Single-bar volume floor (vol_ratio >= threshold)
  - Candle-quality filter (close near high/low of trigger bar)
  - Minimum breakout distance beyond PDL (in ATR)
  - Combinations of the above
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fetch_data import load_csv
import _scalping_strategy_shootout as s

RR = 2.5
ENTRY_START = s.time(9, 30)
ENTRY_END = s.time(13, 0)
PDL_SL_MULT = 1.0

FILTERS = [
    {"name": "Baseline", "vol_min": None, "quality": None, "breakout_atr": 0.10},
    {"name": "Vol>=0.8x", "vol_min": 0.8, "quality": None, "breakout_atr": 0.10},
    {"name": "Vol>=1.0x", "vol_min": 1.0, "quality": None, "breakout_atr": 0.10},
    {"name": "Vol>=1.2x", "vol_min": 1.2, "quality": None, "breakout_atr": 0.10},
    {"name": "BodyClose60", "vol_min": None, "quality": 0.60, "breakout_atr": 0.10},
    {"name": "BodyClose70", "vol_min": None, "quality": 0.70, "breakout_atr": 0.10},
    {"name": "Breakout0.15ATR", "vol_min": None, "quality": None, "breakout_atr": 0.15},
    {"name": "Breakout0.20ATR", "vol_min": None, "quality": None, "breakout_atr": 0.20},
    {"name": "Vol0.8+Body60", "vol_min": 0.8, "quality": 0.60, "breakout_atr": 0.10},
    {"name": "Vol1.0+Body60", "vol_min": 1.0, "quality": 0.60, "breakout_atr": 0.10},
    {"name": "Vol0.8+Break0.15", "vol_min": 0.8, "quality": None, "breakout_atr": 0.15},
    {"name": "Body60+Break0.15", "vol_min": None, "quality": 0.60, "breakout_atr": 0.15},
    {"name": "Vol0.8+Body60+Break0.15", "vol_min": 0.8, "quality": 0.60, "breakout_atr": 0.15},
]


def simulate_prearmed_filtered(sym: str, blocks: list[dict], rr: float, vol_min, quality, breakout_atr):
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
            c = float(row["close"])
            atr = row["atr"]
            if pd.isna(atr) or atr <= 0:
                continue

            if pos is not None:
                if activate_idx is not None and i >= activate_idx:
                    hit, pnl = s.check_exit(row, pos)
                    if hit:
                        trades.append({**pos, "pnl": pnl, "res": hit})
                        if hit == "SL":
                            dsl += 1
                        pos, activate_idx = None, None
                continue

            if not (ENTRY_START <= ts.time() <= ENTRY_END):
                continue
            if cd_until and ts < cd_until:
                continue
            if dsl >= s.MAX_SL_PER_DAY:
                continue

            prev_close = float(rows.iloc[i - 1]["close"]) if i > 0 else None
            if prev_close is None:
                continue

            vol_ratio = row.get("vol_ratio", float("nan"))
            if pd.isna(vol_ratio):
                vsma = row["vol_sma"]
                vol_ratio = row["volume"] / vsma if vsma > 0 and not pd.isna(vsma) else float("nan")
            if vol_min is not None and (pd.isna(vol_ratio) or vol_ratio < vol_min):
                continue

            direction = None
            entry = None
            trigger = float(atr) * breakout_atr
            if ("LONG" not in dirs_used
                    and prev_close <= pdl_h
                    and row["high"] >= pdl_h + trigger
                    and c >= pdl_h):
                direction = "LONG"
                entry = pdl_h + trigger
            elif ("SHORT" not in dirs_used
                  and prev_close >= pdl_l
                  and row["low"] <= pdl_l - trigger
                  and c <= pdl_l):
                direction = "SHORT"
                entry = pdl_l - trigger

            if direction is None:
                continue

            if quality is not None:
                bar_range = float(row["high"] - row["low"])
                if bar_range <= 0:
                    continue
                if direction == "LONG":
                    close_pos = (c - float(row["low"])) / bar_range
                    if close_pos < quality:
                        continue
                else:
                    close_pos = (float(row["high"]) - c) / bar_range
                    if close_pos < quality:
                        continue

            stop_dist = float(atr) * PDL_SL_MULT
            pos = s.open_position(sym, block["date"], direction, float(entry), stop_dist, rr, float(atr), ts)
            dirs_used.add(direction)
            cd_until = ts + s.COOLDOWN
            activate_idx = i + 1

        s.eod_exit(trades, pos, float(rows.iloc[-1]["close"]))
    return trades


def run():
    print("=" * 140)
    print("PREARMED PDL FILTER SWEEP — 90 days, 48 stocks, 3-min candles, no friction")
    print("=" * 140)
    print(f"Window: {ENTRY_START:%H:%M}-{ENTRY_END:%H:%M} | RR={RR}")
    print()

    stock_blocks = {}
    sectors = {}
    for sym, sec in s.NIFTY50:
        p = s.csv_1min(sym)
        if not p.exists():
            continue
        df = load_csv(str(p))
        df3 = s.resample_3min(df)
        ind = s.compute_indicators(df3)
        blocks = s.get_day_blocks(ind)
        if blocks:
            stock_blocks[sym] = blocks
            sectors[sym] = sec

    rows = []
    for cfg in FILTERS:
        all_trades = []
        per_stock = {}
        for sym, blocks in stock_blocks.items():
            trades = simulate_prearmed_filtered(
                sym, blocks, RR,
                cfg["vol_min"], cfg["quality"], cfg["breakout_atr"],
            )
            all_trades.extend(trades)
            per_stock[sym] = sum(t["pnl"] for t in trades)

        ranked = sorted(per_stock.items(), key=lambda x: x[1], reverse=True)
        top10 = {sym for sym, _ in ranked[:10]}
        top10_trades = [t for t in all_trades if t["sym"] in top10]

        all_dates = sorted({t["date"] for t in all_trades})
        recent10_dates = set(all_dates[-10:])
        recent10_all = [t for t in all_trades if t["date"] in recent10_dates]

        top10_dates = sorted({t["date"] for t in top10_trades})
        recent10_top10_dates = set(top10_dates[-10:])
        recent10_top10 = [t for t in top10_trades if t["date"] in recent10_top10_dates]

        a48 = s.analyze(all_trades)
        a10 = s.analyze(top10_trades)
        r10 = s.analyze(recent10_top10)
        rows.append(
            {
                "name": cfg["name"],
                "all48": a48,
                "top10": a10,
                "recent10": r10,
                "ranked": ranked,
            }
        )
        print(
            f"{cfg['name']:<24}  all48=₹{a48['pnl']:>8,.0f}  top10=₹{a10['pnl']:>8,.0f}  "
            f"recent10=₹{r10['pnl']:>7,.0f}  WR={a48['wr']:>5.1f}%  DD=₹{a48['dd']:>7,.0f}"
        )

    rows.sort(key=lambda r: r["top10"]["pnl"], reverse=True)
    print("\n" + "=" * 140)
    print("RANKED BY 90D TOP-10 P&L")
    print("=" * 140)
    print(f"  {'#':>2} {'Filter':<24} {'All48':>10} {'Top10':>10} {'Recent10':>10} {'WR%':>6} {'DD':>10} {'PF':>5} {'Avg':>7}")
    print("  " + "─" * 110)
    for i, r in enumerate(rows, 1):
        a48 = r["all48"]
        a10 = r["top10"]
        r10 = r["recent10"]
        print(
            f"  {i:>2} {r['name']:<24} ₹{a48['pnl']:>8,.0f} ₹{a10['pnl']:>8,.0f} ₹{r10['pnl']:>8,.0f} "
            f"{a48['wr']:>5.1f}% ₹{a48['dd']:>8,.0f} {a48['pf']:>5.2f} ₹{a48['avg']:>5,.0f}"
        )

    recent_rank = sorted(rows, key=lambda r: r["recent10"]["pnl"], reverse=True)
    print("\nTop by recent 10-day top-10 P&L:")
    for i, r in enumerate(recent_rank[:5], 1):
        print(
            f"  {i:>2}. {r['name']:<24} recent10=₹{r['recent10']['pnl']:>8,.0f}  "
            f"90d top10=₹{r['top10']['pnl']:>8,.0f}"
        )

    winner = rows[0]
    print("\n" + "=" * 140)
    print(f"WINNER — {winner['name']}")
    print("=" * 140)
    print(f"All 48 P&L: ₹{winner['all48']['pnl']:,.0f}")
    print(f"Top 10 P&L: ₹{winner['top10']['pnl']:,.0f}")
    print(f"Recent 10-day Top 10 P&L: ₹{winner['recent10']['pnl']:,.0f}")
    print(f"WR: {winner['all48']['wr']:.1f}% | DD: ₹{winner['all48']['dd']:,.0f} | PF: {winner['all48']['pf']:.2f}")

    print("\nTop 10 stocks for winner:")
    for i, (sym, pnl) in enumerate(winner["ranked"][:10], 1):
        print(f"  {i:>2}. {sym:<14} {sectors[sym]:<12} ₹{pnl:>8,.0f}")


if __name__ == "__main__":
    run()

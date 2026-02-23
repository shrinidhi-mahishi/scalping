#!/usr/bin/env python3
"""
Verify that every per-stock configuration in live_signals.py
produces a P&L improvement over the raw baseline (no HTF, no stagnation).

Tests 4 configs per stock:
  1. BASE        — no HTF, no stagnation
  2. HTF         — 15-min HTF filter only
  3. BASE+STAG   — stagnation exit only
  4. HTF+STAG    — HTF + stagnation

Compares the config CURRENTLY assigned in live_signals.py against BASE
and highlights any regressions.
"""

import os
import sys

import backtrader as bt
import backtrader.analyzers as bta
import pandas as pd

from fetch_data import load_csv
from strategy import IndianBrokerCommission, IntradayScalpingStrategy
from strategy_early_exit import EarlyExitStrategy

STOCKS = [
    ("HDFCLIFE", "Finance"),
    ("TATASTEEL", "Metals"),
    ("TITAN", "Consumer"),
    ("M&M", "Auto"),
    ("ADANIPORTS", "Conglomerate"),
    ("ONGC", "Energy"),
    ("BAJAJFINSV", "Finance"),
    ("HINDALCO", "Metals"),
    ("SBILIFE", "Finance"),
]

HTF_STOCKS = {"TATASTEEL", "M&M", "ONGC", "BAJAJFINSV", "HINDALCO", "SBILIFE"}
STAGNATION_SKIP = {"HDFCLIFE", "TITAN", "ONGC"}

CASH = 50_000.0

CONFIGS = [
    ("BASE",     False, IntradayScalpingStrategy, {}),
    ("HTF",      True,  IntradayScalpingStrategy, {}),
    ("BASE+S",   False, EarlyExitStrategy, {
        "exit_mode": "stagnation", "max_trade_bars": 8, "min_profit_atr": 0.2,
    }),
    ("HTF+S",    True,  EarlyExitStrategy, {
        "exit_mode": "stagnation", "max_trade_bars": 8, "min_profit_atr": 0.2,
    }),
]


def csv_path(symbol: str) -> str:
    return f"data/{symbol.replace('&', '')}_3min.csv"


def run_one(df, use_htf, strat_cls, strat_kw):
    cerebro = bt.Cerebro()
    feed_kw = dict(
        dataname=df, datetime=None,
        open="open", high="high", low="low", close="close",
        volume="volume", openinterest=-1,
    )
    cerebro.adddata(bt.feeds.PandasData(**feed_kw))
    if use_htf:
        cerebro.resampledata(
            bt.feeds.PandasData(**feed_kw),
            timeframe=bt.TimeFrame.Minutes, compression=15,
        )
    kw = {"printlog": False, "use_htf_filter": use_htf}
    if strat_kw:
        kw.update(strat_kw)
    cerebro.addstrategy(strat_cls, **kw)
    cerebro.broker.setcash(CASH)
    cerebro.broker.addcommissioninfo(IndianBrokerCommission())
    cerebro.broker.set_checksubmit(False)
    cerebro.broker.set_slippage_perc(
        perc=0.0001, slip_open=True, slip_limit=False,
        slip_match=True, slip_out=False,
    )
    cerebro.addanalyzer(bta.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bta.DrawDown, _name="drawdown")
    results = cerebro.run()
    strat = results[0]

    ta = strat.analyzers.trades.get_analysis()
    dd = strat.analyzers.drawdown.get_analysis()
    total = ta.get("total", {}).get("total", 0)
    won = ta.get("won", {}).get("total", 0)
    avg_win = ta.get("won", {}).get("pnl", {}).get("average", 0) if won else 0
    lost = ta.get("lost", {}).get("total", 0)
    avg_loss = ta.get("lost", {}).get("pnl", {}).get("average", 0) if lost else 0
    net_pnl = cerebro.broker.getvalue() - CASH
    wr = won / total * 100 if total else 0
    pf = abs(avg_win * won) / abs(avg_loss * lost) if lost and avg_loss else 0
    max_dd = dd.get("max", {}).get("drawdown", 0)

    return {
        "total": total, "won": won, "wr": wr,
        "net_pnl": net_pnl, "pf": pf, "max_dd": max_dd,
    }


def live_config(sym):
    """What config does live_signals.py use for this stock?"""
    has_htf = sym in HTF_STOCKS
    has_stag = sym not in STAGNATION_SKIP
    if has_htf and has_stag:
        return "HTF+S"
    elif has_htf:
        return "HTF"
    elif has_stag:
        return "BASE+S"
    else:
        return "BASE"


def main():
    W = 120
    print(f"\n{'=' * W}")
    print(f"  VERIFICATION: Does each stock's live config beat the raw baseline?")
    print(f"{'=' * W}\n")

    print(
        f"  {'Stock':<12} {'Live':>6}  "
        f"{'BASE P&L':>10} {'Live P&L':>10} {'Delta':>10}  "
        f"{'BASE Tr':>7} {'Live Tr':>7}  "
        f"{'BASE Win%':>9} {'Live Win%':>9}  "
        f"{'BASE PF':>7} {'Live PF':>7}  "
        f"{'Verdict':>8}"
    )
    print(f"  {'-' * (W - 4)}")

    total_base = 0.0
    total_live = 0.0
    regressions = []

    for sym, sec in STOCKS:
        path = csv_path(sym)
        if not os.path.exists(path):
            print(f"  {sym:<12}  SKIP — no data")
            continue

        df = load_csv(path)
        results = {}
        for name, use_htf, strat_cls, strat_kw in CONFIGS:
            results[name] = run_one(df, use_htf, strat_cls, strat_kw)

        lc = live_config(sym)
        base = results["BASE"]
        live = results[lc]
        delta = live["net_pnl"] - base["net_pnl"]
        total_base += base["net_pnl"]
        total_live += live["net_pnl"]

        verdict = "OK" if delta >= 0 else "REGRESS"
        v_color = "" if delta >= 0 else " <<<<<"

        print(
            f"  {sym:<12} {lc:>6}  "
            f"{base['net_pnl']:>+10,.2f} {live['net_pnl']:>+10,.2f} {delta:>+10,.2f}  "
            f"{base['total']:>7} {live['total']:>7}  "
            f"{base['wr']:>8.1f}% {live['wr']:>8.1f}%  "
            f"{base['pf']:>7.2f} {live['pf']:>7.2f}  "
            f"{verdict:>8}{v_color}"
        )

        if delta < 0:
            regressions.append({
                "sym": sym, "config": lc, "delta": delta,
                "all": results,
            })

    print(f"  {'-' * (W - 4)}")
    total_delta = total_live - total_base
    print(
        f"  {'TOTAL':<12} {'':>6}  "
        f"{total_base:>+10,.2f} {total_live:>+10,.2f} {total_delta:>+10,.2f}"
    )

    # Show all 4 configs for any regressions
    if regressions:
        print(f"\n{'=' * W}")
        print(f"  REGRESSIONS — All 4 configs for affected stocks")
        print(f"{'=' * W}")
        for r in regressions:
            sym = r["sym"]
            res = r["all"]
            print(f"\n  {sym}  (currently: {r['config']}, delta: {r['delta']:+,.2f})")
            print(
                f"    {'Config':<8} {'Trades':>7} {'Won':>4} {'Win%':>6} "
                f"{'Net P&L':>12} {'PF':>6} {'MaxDD%':>7}  {'vs BASE':>10}"
            )
            for name in ("BASE", "HTF", "BASE+S", "HTF+S"):
                m = res[name]
                d = m["net_pnl"] - res["BASE"]["net_pnl"]
                best = " ★" if m["net_pnl"] == max(
                    res[n]["net_pnl"] for n in res
                ) else ""
                print(
                    f"    {name:<8} {m['total']:>7} {m['won']:>4} {m['wr']:>5.1f}% "
                    f"{m['net_pnl']:>+12,.2f} {m['pf']:>6.2f} {m['max_dd']:>6.2f}%  "
                    f"{d:>+10,.2f}{best}"
                )

        print(f"\n  RECOMMENDED FIXES:")
        for r in regressions:
            res = r["all"]
            best_name = max(res, key=lambda k: res[k]["net_pnl"])
            best_pnl = res[best_name]["net_pnl"]
            print(
                f"    {r['sym']}: switch from {r['config']} → {best_name} "
                f"(P&L: {best_pnl:+,.2f})"
            )
    else:
        print(f"\n  ✓ ALL CONFIGURATIONS VERIFIED — every change improves P&L over raw baseline")

    print()


if __name__ == "__main__":
    main()

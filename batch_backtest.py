#!/usr/bin/env python3
"""
Multi-stock batch backtester with detailed per-trade reporting.

Usage:
  python batch_backtest.py              # fetch from Angel One + backtest
  python batch_backtest.py --cached     # use cached CSVs only (no API calls)
"""

import argparse
import os
import sys
import time as _time

import backtrader as bt
import backtrader.analyzers as bta
import pandas as pd

from fetch_data import AngelOneClient, load_csv, save_csv
from strategy import IndianBrokerCommission, IntradayScalpingStrategy

# ─── Configuration ────────────────────────────────────────────────────────────

STOCKS = [
    ("MARUTI", "Auto"),
    ("HDFCLIFE", "Finance"),
    ("TATASTEEL", "Metals"),
    ("TITAN", "Consumer"),
    ("ONGC", "Energy"),
]

CASH = 50_000.0
FETCH_DAYS = 90


def csv_path(symbol: str) -> str:
    return f"data/{symbol.replace('&', '')}_3min.csv"


# ─── Data Fetching ────────────────────────────────────────────────────────────


def fetch_all(cached_only: bool = False) -> dict[str, pd.DataFrame]:
    os.makedirs("data", exist_ok=True)
    data: dict[str, pd.DataFrame] = {}
    client = None

    for symbol, _ in STOCKS:
        path = csv_path(symbol)

        if os.path.exists(path):
            print(f"  {symbol:<12}  cached  -> loading {path}")
            data[symbol] = load_csv(path)
            continue

        if cached_only:
            print(f"  {symbol:<12}  SKIP    -> no CSV found")
            continue

        if client is None:
            client = AngelOneClient()
            client.connect()

        try:
            print(f"  {symbol:<12}  fetch   -> {FETCH_DAYS}d from Angel One ...", end="", flush=True)
            df = client.fetch_history(symbol, days=FETCH_DAYS)
            save_csv(df, path)
            data[symbol] = df
            print(f"  {len(df)} candles")
            _time.sleep(3)
        except Exception as e:
            print(f"  FAILED: {e}")

    return data


# ─── Backtest Engine ──────────────────────────────────────────────────────────


HTF_STOCKS = {"MARUTI", "HDFCLIFE", "TATASTEEL", "ONGC"}
REGIME_STOCKS: set[str] = set()


def run_one(df: pd.DataFrame, symbol: str = "") -> tuple:
    cerebro = bt.Cerebro()
    feed_kw = dict(
        dataname=df, datetime=None,
        open="open", high="high", low="low", close="close",
        volume="volume", openinterest=-1,
    )
    cerebro.adddata(bt.feeds.PandasData(**feed_kw))
    cerebro.resampledata(
        bt.feeds.PandasData(**feed_kw),
        timeframe=bt.TimeFrame.Minutes, compression=15,
    )
    cerebro.addstrategy(
        IntradayScalpingStrategy, printlog=False,
        use_htf_filter=symbol in HTF_STOCKS,
        use_regime_filter=symbol in REGIME_STOCKS,
    )
    cerebro.broker.setcash(CASH)
    cerebro.broker.addcommissioninfo(IndianBrokerCommission())
    cerebro.broker.set_checksubmit(False)
    cerebro.broker.set_slippage_perc(
        perc=0.0001, slip_open=True, slip_limit=False,
        slip_match=True, slip_out=False,
    )
    cerebro.addanalyzer(bta.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bta.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bta.SharpeRatio, _name="sharpe", riskfreerate=0.065 / 252)

    results = cerebro.run()
    return cerebro, results[0]


# ─── Reporting ────────────────────────────────────────────────────────────────

W = 100  # report width


def fmt_pnl(v: float) -> str:
    sign = "+" if v > 0 else ""
    return f"{sign}{v:,.2f}"


def print_stock_report(symbol: str, sector: str, df: pd.DataFrame, cerebro, strat) -> dict:
    trades = strat._trade_log
    ta = strat.analyzers.trades.get_analysis()
    dd = strat.analyzers.drawdown.get_analysis()
    sharpe = strat.analyzers.sharpe.get_analysis()

    net_pnl = cerebro.broker.getvalue() - CASH
    total = ta.get("total", {}).get("total", 0)
    won = ta.get("won", {}).get("total", 0)
    lost = ta.get("lost", {}).get("total", 0)
    win_rate = won / total * 100 if total else 0.0
    avg_win = ta.get("won", {}).get("pnl", {}).get("average", 0) if won else 0.0
    avg_loss = ta.get("lost", {}).get("pnl", {}).get("average", 0) if lost else 0.0
    max_dd = dd.get("max", {}).get("drawdown", 0)
    sr = sharpe.get("sharperatio")
    pf = abs(avg_win * won) / abs(avg_loss * lost) if lost and avg_loss else 0.0

    longs = [t for t in trades if t.get("direction") == "LONG"]
    shorts = [t for t in trades if t.get("direction") == "SHORT"]
    long_wins = sum(1 for t in longs if t.get("pnlcomm", 0) > 0)
    short_wins = sum(1 for t in shorts if t.get("pnlcomm", 0) > 0)
    long_pnl = sum(t.get("pnlcomm", 0) for t in longs)
    short_pnl = sum(t.get("pnlcomm", 0) for t in shorts)

    # ── Header ──
    print(f"\n{'=' * W}")
    print(f"  {symbol}  |  {sector}  |  {df.index[0]:%Y-%m-%d} to {df.index[-1]:%Y-%m-%d}  |  {len(df):,} candles")
    print(f"{'=' * W}")

    # ── Trade log table ──
    if trades:
        hdr = f"  {'#':>3}  {'Call':<6}  {'Entry Time':<18}  {'Entry':>10}  {'Exit Time':<18}  {'Exit':>10}  {'Reason':<7}  {'SL':>9}  {'TP':>9}  {'Net P&L':>10}  {'RSI':>5}"
        print(hdr)
        print(f"  {'-' * (W - 4)}")

        for i, t in enumerate(trades, 1):
            d = t.get("direction", "?")
            et = t.get("entry_time")
            ep = t.get("entry_price", 0)
            xt = t.get("exit_time")
            xp = t.get("exit_price", 0)
            xr = t.get("exit_reason", "?")
            sl = t.get("sl_price", 0)
            tp = t.get("tp_price", 0)
            pnlc = t.get("pnlcomm", 0)
            rsi = t.get("signal_rsi", 0)

            et_s = f"{et:%Y-%m-%d %H:%M}" if et else "--"
            xt_s = f"{xt:%Y-%m-%d %H:%M}" if xt else "--"
            pnl_s = fmt_pnl(pnlc)

            print(
                f"  {i:>3}  {d:<6}  {et_s:<18}  {ep:>10,.2f}  {xt_s:<18}  {xp:>10,.2f}"
                f"  {xr:<7}  {sl:>9,.2f}  {tp:>9,.2f}  {pnl_s:>10}  {rsi:>5.1f}"
            )

    # ── Per-stock summary ──
    best = max((t.get("pnlcomm", 0) for t in trades), default=0)
    worst = min((t.get("pnlcomm", 0) for t in trades), default=0)

    print(f"\n  {'-' * (W - 4)}")
    print(f"  {'SUMMARY':^{W - 4}}")
    print(f"  {'-' * (W - 4)}")
    col = 44
    l1 = f"  {'Total Trades':<20}: {total:<{col}}{'Net P&L':<16}: {fmt_pnl(net_pnl)}"
    l2 = f"  {'Won / Lost':<20}: {won}W / {lost}L{'':<{col - len(f'{won}W / {lost}L')}}{'Win Rate':<16}: {win_rate:.1f}%"
    l3 = f"  {'Avg Win':<20}: {fmt_pnl(avg_win):<{col}}{'Avg Loss':<16}: {fmt_pnl(avg_loss)}"
    l4 = f"  {'Best Trade':<20}: {fmt_pnl(best):<{col}}{'Worst Trade':<16}: {fmt_pnl(worst)}"
    l5 = f"  {'Profit Factor':<20}: {pf:.2f}{'':<{col - len(f'{pf:.2f}')}}{'Max Drawdown':<16}: {max_dd:.2f}%"
    sr_s = f"{sr:.3f}" if sr is not None else "N/A"
    l6 = f"  {'Sharpe Ratio':<20}: {sr_s}"
    print(l1)
    print(l2)
    print(l3)
    print(l4)
    print(l5)
    print(l6)

    print(f"\n  LONG  : {len(longs):>3} trades ({long_wins}W)  Net = {fmt_pnl(long_pnl)}")
    print(f"  SHORT : {len(shorts):>3} trades ({short_wins}W)  Net = {fmt_pnl(short_pnl)}")
    print(f"  {'=' * (W - 4)}")

    return {
        "symbol": symbol,
        "sector": sector,
        "total": total,
        "won": won,
        "lost": lost,
        "win_rate": win_rate,
        "net_pnl": net_pnl,
        "pf": pf,
        "max_dd": max_dd,
        "sharpe": sr,
        "longs": len(longs),
        "shorts": len(shorts),
        "long_pnl": long_pnl,
        "short_pnl": short_pnl,
        "long_wins": long_wins,
        "short_wins": short_wins,
        "best": best,
        "worst": worst,
    }


def print_comparison(summaries: list[dict]) -> None:
    summaries.sort(key=lambda s: s["net_pnl"], reverse=True)

    print(f"\n{'=' * W}")
    print(f"  COMPARATIVE SUMMARY — ALL STOCKS (Ranked by Net P&L)")
    print(f"{'=' * W}")

    hdr = (
        f"  {'Symbol':<12} {'Sector':<14} {'Trades':>6} {'Win%':>6} "
        f"{'Net P&L':>12} {'MaxDD%':>7} {'PF':>6} "
        f"{'Longs':>6} {'Shorts':>7} {'Best Call':<10} {'Verdict':<10}"
    )
    print(hdr)
    print(f"  {'-' * (W - 4)}")

    total_pnl = 0.0
    total_trades = 0

    for s in summaries:
        if s["total"] == 0:
            verdict = "--"
            best_call = "--"
        else:
            if s["pf"] >= 1.0 and s["win_rate"] >= 45:
                verdict = "TRADEABLE"
            elif s["pf"] >= 0.7:
                verdict = "MARGINAL"
            else:
                verdict = "AVOID"
            best_call = "LONG" if s["long_pnl"] >= s["short_pnl"] else "SHORT"

        pnl_s = f"{fmt_pnl(s['net_pnl']):>12}"
        print(
            f"  {s['symbol']:<12} {s['sector']:<14} {s['total']:>6} {s['win_rate']:>5.1f}% "
            f"{pnl_s} {s['max_dd']:>6.2f}% {s['pf']:>6.2f} "
            f"{s['longs']:>6} {s['shorts']:>7} {best_call:<10} {verdict:<10}"
        )
        total_pnl += s["net_pnl"]
        total_trades += s["total"]

    print(f"  {'-' * (W - 4)}")
    print(
        f"  {'PORTFOLIO':<12} {'—':<14} {total_trades:>6} {'':>6} "
        f"{fmt_pnl(total_pnl):>12}"
    )

    # ── Sector breakdown ──
    sectors: dict[str, list] = {}
    for s in summaries:
        sectors.setdefault(s["sector"], []).append(s)

    print(f"\n  SECTOR BREAKDOWN:")
    print(f"  {'-' * 60}")
    for sector, items in sectors.items():
        sec_pnl = sum(s["net_pnl"] for s in items)
        sec_trades = sum(s["total"] for s in items)
        sec_wr = (
            sum(s["won"] for s in items) / sec_trades * 100 if sec_trades else 0
        )
        print(
            f"  {sector:<16} {sec_trades:>3} trades  |  "
            f"Win Rate: {sec_wr:>5.1f}%  |  Net P&L: {fmt_pnl(sec_pnl):>12}"
        )

    print(f"\n{'=' * W}")


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Batch backtest across NSE stocks")
    parser.add_argument("--cached", action="store_true", help="Only use cached CSVs")
    args = parser.parse_args()

    header = (
        f"  INTRADAY SCALPING BACKTEST v3 — {len(STOCKS)} STOCK SHORTLIST + HTF\n"
        f"  Capital: {CASH:,.0f}  |  Sizing: 1% risk/ATR  |  RR 1:1.5  |  Brokerage: min(0.03%,Rs20)/leg\n"
        f"  Entry: 10:00-12:00 & 14:00-15:00  |  Squareoff: 15:15\n"
        f"  HTF: 15-min 21-EMA trend filter (all stocks)  |  Slippage: 0.01%"
    )
    print(f"\n{'=' * W}")
    print(header)
    print(f"{'=' * W}")

    # ── Fetch ──
    print(f"\n[1/3] FETCHING DATA ...\n")
    all_data = fetch_all(cached_only=args.cached)
    loaded = len(all_data)
    print(f"\n  -> {loaded}/{len(STOCKS)} stocks loaded\n")

    if loaded == 0:
        print("No data available. Run without --cached to fetch from Angel One.")
        sys.exit(1)

    # ── Backtest ──
    print(f"[2/3] RUNNING BACKTESTS ...\n")
    results: dict[str, tuple] = {}
    for symbol, sector in STOCKS:
        if symbol not in all_data:
            continue
        print(f"  {symbol:<12} backtesting ...", end="", flush=True)
        cerebro, strat = run_one(all_data[symbol], symbol)
        results[symbol] = (cerebro, strat)
        n = len(strat._trade_log)
        print(f"  {n} trades")

    # ── Reports ──
    print(f"\n[3/3] DETAILED REPORTS")
    summaries: list[dict] = []
    for symbol, sector in STOCKS:
        if symbol not in results:
            continue
        cerebro, strat = results[symbol]
        summary = print_stock_report(symbol, sector, all_data[symbol], cerebro, strat)
        summaries.append(summary)

    print_comparison(summaries)


if __name__ == "__main__":
    main()

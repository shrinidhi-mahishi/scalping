#!/usr/bin/env python3
"""
Intraday Scalping Backtest Runner (v2)

Usage:
  python run_backtest.py --sample                      # generated sample data
  python run_backtest.py --csv data/reliance.csv       # from CSV
  python run_backtest.py --fetch RELIANCE --days 30    # live Angel One data
"""

import argparse
import sys

import backtrader as bt
import backtrader.analyzers as bta
import pandas as pd

from fetch_data import AngelOneClient, generate_sample_data, load_csv, save_csv
from strategy import IndianBrokerCommission, IntradayScalpingStrategy


# ─── Data Feed ────────────────────────────────────────────────────────────────


def make_data_feed(df: pd.DataFrame) -> bt.feeds.PandasData:
    return bt.feeds.PandasData(
        dataname=df,
        datetime=None,
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
        openinterest=-1,
    )


# ─── Performance Report ──────────────────────────────────────────────────────


def print_report(cerebro: bt.Cerebro, strat) -> None:
    start_val = cerebro.broker.startingcash
    end_val = cerebro.broker.getvalue()
    ret_pct = (end_val - start_val) / start_val * 100

    sep = "=" * 60
    print(f"\n{sep}")
    print("  BACKTEST PERFORMANCE REPORT (v2)")
    print(f"  Sizing: 1% risk / ATR  |  Brokerage: min(0.03%,Rs20)/leg")
    print(sep)
    print(f"  Starting Capital : ₹{start_val:>14,.2f}")
    print(f"  Ending Capital   : ₹{end_val:>14,.2f}")
    print(f"  Net P&L          : ₹{end_val - start_val:>14,.2f}")
    print(f"  Total Return     : {ret_pct:>14.2f}%")

    ta = strat.analyzers.trades.get_analysis()
    total = ta.get("total", {}).get("total", 0)

    if total > 0:
        won = ta.get("won", {}).get("total", 0)
        lost = ta.get("lost", {}).get("total", 0)
        win_rate = won / total * 100

        avg_win = ta.get("won", {}).get("pnl", {}).get("average", 0)
        avg_loss = ta.get("lost", {}).get("pnl", {}).get("average", 0)

        print(f"\n  Total Trades     : {total:>14}")
        print(f"  Wins             : {won:>14}")
        print(f"  Losses           : {lost:>14}")
        print(f"  Win Rate         : {win_rate:>14.1f}%")
        print(f"  Avg Win          : ₹{avg_win:>14,.2f}")
        print(f"  Avg Loss         : ₹{avg_loss:>14,.2f}")

        if lost > 0 and avg_loss != 0:
            pf = abs(avg_win * won) / abs(avg_loss * lost)
            print(f"  Profit Factor    : {pf:>14.2f}")

        longest = ta.get("len", {}).get("max", "—")
        print(f"  Longest Trade    : {str(longest) + ' bars':>14}")

    dd = strat.analyzers.drawdown.get_analysis()
    max_dd = dd.get("max", {}).get("drawdown", 0)
    max_dd_money = dd.get("max", {}).get("moneydown", 0)
    print(f"\n  Max Drawdown     : {max_dd:>14.2f}%")
    print(f"  Max Drawdown ₹   : ₹{max_dd_money:>14,.2f}")

    sharpe = strat.analyzers.sharpe.get_analysis()
    sr = sharpe.get("sharperatio")
    sr_str = f"{sr:.3f}" if sr is not None else "N/A"
    print(f"  Sharpe Ratio     : {sr_str:>14}")

    print(sep)


# ─── CLI ──────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="NSE Intraday Scalping Backtest (v2)",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--sample", action="store_true", help="Use generated sample data")
    src.add_argument("--csv", type=str, help="Path to OHLCV CSV file")
    src.add_argument("--fetch", type=str, help="Fetch from Angel One (symbol name)")

    p.add_argument("--days", type=int, default=30, help="Trading days (default: 30)")
    p.add_argument(
        "--cash", type=float, default=100_000.0, help="Starting capital (default: Rs 1,00,000)"
    )
    p.add_argument("--fixed-qty", type=int, default=0,
                   help="Override dynamic sizing with fixed qty (default: 0 = dynamic)")
    p.add_argument("--plot", action="store_true", help="Show chart after backtest")
    p.add_argument("--save-csv", type=str, help="Save data to CSV for reuse")
    p.add_argument("--quiet", action="store_true", help="Suppress trade-by-trade logs")

    return p.parse_args()


def main():
    args = parse_args()

    if args.sample:
        print(f"[*] Generating sample 3-min data ({args.days} trading days) ...")
        df = generate_sample_data(days=args.days)
    elif args.csv:
        print(f"[*] Loading from {args.csv} ...")
        df = load_csv(args.csv)
    else:
        print(f"[*] Fetching {args.fetch} from Angel One ({args.days} days) ...")
        client = AngelOneClient()
        df = client.fetch_history(args.fetch, days=args.days)

    if args.save_csv:
        save_csv(df, args.save_csv)

    print(
        f"[*] {len(df)} candles  |  {df.index[0]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d}"
    )

    cerebro = bt.Cerebro()

    cerebro.adddata(make_data_feed(df))
    cerebro.addstrategy(
        IntradayScalpingStrategy,
        trade_qty=args.fixed_qty,
        printlog=not args.quiet,
    )

    cerebro.broker.setcash(args.cash)
    cerebro.broker.addcommissioninfo(IndianBrokerCommission())
    cerebro.broker.set_checksubmit(False)

    cerebro.addanalyzer(bta.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bta.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bta.SharpeRatio, _name="sharpe", riskfreerate=0.065 / 252)
    cerebro.addanalyzer(bta.Returns, _name="returns")

    sizing = f"Qty={args.fixed_qty}" if args.fixed_qty else "1% risk/ATR"
    print(f"[*] Running backtest  |  Capital=₹{args.cash:,.0f}  {sizing}\n")
    results = cerebro.run()
    strat = results[0]

    print_report(cerebro, strat)

    if args.plot:
        cerebro.plot(style="candlestick", volume=True)


if __name__ == "__main__":
    main()

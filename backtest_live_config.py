#!/usr/bin/env python3
"""
Backtest the exact live_signals.py configuration on the 9 shortlisted stocks.

Combines:
  - EnhancedStrategy filters (VWAP extension 1.5x ATR, tighter RSI 50-75/25-50)
  - vol_mult = 1.5 (raised minimum volume)
  - Stagnation exit (8 bars, < 0.2x ATR profit, with per-stock skip list)
  - HTF 15-min 21-EMA filter (per stock)
  - Regime filter (daily 21-EMA)

Usage:
  python backtest_live_config.py                  # all 9 stocks, full data
  python backtest_live_config.py --days 90        # last 90 trading days
  python backtest_live_config.py --stocks TITAN ONGC
"""

import argparse
import os
import sys

import backtrader as bt
import backtrader.analyzers as bta
import pandas as pd

from fetch_data import load_csv
from strategy import IndianBrokerCommission
from strategy_enhanced import EnhancedStrategy

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

LIVE_PARAMS = dict(
    vol_mult=1.5,
    vwap_ext_atr=1.5,
    long_rsi_low=50,
    long_rsi_high=75,
    short_rsi_low=25,
    short_rsi_high=50,
    use_regime_filter=True,
    printlog=False,
)

STAG_MAX_BARS = 8
STAG_MIN_PROFIT_ATR = 0.2


class LiveConfigStrategy(EnhancedStrategy):
    """EnhancedStrategy + stagnation exit (matching live_signals.py exactly)."""

    params = (
        ("stag_max_bars", STAG_MAX_BARS),
        ("stag_min_profit_atr", STAG_MIN_PROFIT_ATR),
        ("use_stagnation", True),
    )

    def __init__(self):
        super().__init__()
        self._stag_entry_bar = 0

    def notify_order(self, order):
        was_entry = (
            self._main_ref is not None
            and order.ref == self._main_ref
            and order.status == order.Completed
        )
        super().notify_order(order)
        if was_entry:
            self._stag_entry_bar = len(self)

    def next(self):
        if (
            self.p.use_stagnation
            and self.position
            and not self._order_pending
            and not self._is_squareoff()
            and self._stag_entry_bar > 0
            and self._active_trade
        ):
            bars = len(self) - self._stag_entry_bar
            if bars >= self.p.stag_max_bars:
                entry_price = self._active_trade.get("entry_price", 0)
                if entry_price > 0:
                    is_long = self.position.size > 0
                    c = self.data.close[0]
                    unrealized = (c - entry_price) if is_long else (entry_price - c)
                    if unrealized < self.p.stag_min_profit_atr * self._signal_atr:
                        self._cancel_bracket()
                        if self._active_trade is not None:
                            self._active_trade["exit_reason"] = "STAGNATION"
                        self.close()
                        return

        super().next()


def csv_path(symbol: str) -> str:
    return f"data/{symbol.replace('&', '')}_3min.csv"


def trim_to_days(df: pd.DataFrame, days: int) -> pd.DataFrame:
    """Keep only the last N trading days of data."""
    dates = sorted(df.index.normalize().unique())
    if len(dates) <= days:
        return df
    cutoff = dates[-days]
    return df[df.index >= cutoff]


def run_stock(symbol: str, df: pd.DataFrame) -> dict:
    use_htf = symbol in HTF_STOCKS
    use_stag = symbol not in STAGNATION_SKIP

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
        LiveConfigStrategy,
        use_htf_filter=use_htf,
        use_stagnation=use_stag,
        **LIVE_PARAMS,
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

    results = cerebro.run()
    strat = results[0]
    return _extract(cerebro, strat, symbol)


def _extract(cerebro, strat, symbol: str) -> dict:
    ta = strat.analyzers.trades.get_analysis()
    dd = strat.analyzers.drawdown.get_analysis()

    total = ta.get("total", {}).get("total", 0)
    won = ta.get("won", {}).get("total", 0)
    lost = ta.get("lost", {}).get("total", 0)
    avg_win = ta.get("won", {}).get("pnl", {}).get("average", 0) if won else 0
    avg_loss = ta.get("lost", {}).get("pnl", {}).get("average", 0) if lost else 0

    win_rate = won / total * 100 if total else 0.0
    pf = abs(avg_win * won) / abs(avg_loss * lost) if lost and avg_loss else 0.0
    net_pnl = cerebro.broker.getvalue() - CASH
    max_dd = dd.get("max", {}).get("drawdown", 0)

    tlog = strat._trade_log
    exit_counts: dict[str, int] = {}
    for t in tlog:
        r = t.get("exit_reason", "UNKNOWN")
        exit_counts[r] = exit_counts.get(r, 0) + 1

    return {
        "symbol": symbol,
        "trades": total,
        "won": won,
        "lost": lost,
        "win_rate": win_rate,
        "pf": pf,
        "net_pnl": net_pnl,
        "max_dd": max_dd,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "exits": exit_counts,
        "htf": symbol in HTF_STOCKS,
        "stag": symbol not in STAGNATION_SKIP,
    }


def main():
    parser = argparse.ArgumentParser(description="Backtest live_signals.py config")
    parser.add_argument("--days", type=int, default=0, help="Limit to last N trading days (0=all)")
    parser.add_argument("--stocks", nargs="*", help="Run only these stocks")
    args = parser.parse_args()

    print("=" * 72)
    print("  BACKTEST — live_signals.py configuration")
    print(f"  Capital: ₹{CASH:,.0f}  |  RR 1:1.5  |  Risk 1%")
    print(f"  Filters: VWAP ext ±1.5 ATR | Vol > 1.5x SMA | RSI 50-75/25-50")
    print(f"  Exits  : Stagnation (8 bars, <0.2x ATR) skip {STAGNATION_SKIP}")
    if args.days:
        print(f"  Period : last {args.days} trading days")
    print("=" * 72)
    print()

    all_results = []

    for symbol, sector in STOCKS:
        if args.stocks and symbol not in args.stocks:
            continue

        path = csv_path(symbol)
        if not os.path.exists(path):
            print(f"  {symbol:<12}  SKIP — no data at {path}")
            continue

        df = load_csv(path)
        if args.days:
            df = trim_to_days(df, args.days)

        dates = sorted(df.index.normalize().unique())
        htf_tag = "HTF" if symbol in HTF_STOCKS else "   "
        stag_tag = "STAG" if symbol not in STAGNATION_SKIP else "    "
        print(f"  {symbol:<12}  {len(df):>5} bars  {len(dates):>3}d  [{htf_tag}] [{stag_tag}]")

        result = run_stock(symbol, df)
        all_results.append(result)

    if not all_results:
        print("\nNo data to backtest.")
        return

    print()
    print("=" * 72)
    print(f"  {'STOCK':<12} {'Trades':>6} {'Won':>5} {'Lost':>5} {'Win%':>6} "
          f"{'PF':>5} {'Net P&L':>10} {'MaxDD%':>7} {'Exits'}")
    print("-" * 72)

    total_pnl = 0.0
    total_trades = 0
    total_won = 0
    total_lost = 0

    for r in all_results:
        exits_str = "  ".join(f"{k}:{v}" for k, v in sorted(r["exits"].items()))
        pnl_str = f"₹{r['net_pnl']:>+,.0f}"
        marker = "✓" if r["net_pnl"] > 0 else "✗"

        print(f"  {r['symbol']:<12} {r['trades']:>6} {r['won']:>5} {r['lost']:>5} "
              f"{r['win_rate']:>5.1f}% {r['pf']:>5.2f} {pnl_str:>10} {r['max_dd']:>6.1f}% "
              f"{marker}  {exits_str}")

        total_pnl += r["net_pnl"]
        total_trades += r["trades"]
        total_won += r["won"]
        total_lost += r["lost"]

    overall_wr = total_won / total_trades * 100 if total_trades else 0

    print("-" * 72)
    pnl_color = "\033[92m" if total_pnl > 0 else "\033[91m"
    rst = "\033[0m"
    print(f"  {'TOTAL':<12} {total_trades:>6} {total_won:>5} {total_lost:>5} "
          f"{overall_wr:>5.1f}%       {pnl_color}₹{total_pnl:>+,.0f}{rst}")
    print("=" * 72)


if __name__ == "__main__":
    main()

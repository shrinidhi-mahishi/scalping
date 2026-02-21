#!/usr/bin/env python3
"""
Generate CSV + text reports from cached backtest data (v2).
Outputs:
  reports/backtest_report.txt   — full console report
  reports/all_trades.csv        — every trade across all stocks
  reports/stock_summary.csv     — per-stock summary metrics
"""

import os
import sys

import backtrader as bt
import backtrader.analyzers as bta
import pandas as pd

from fetch_data import load_csv
from strategy import IndianBrokerCommission, IntradayScalpingStrategy

STOCKS = [
    ("MARUTI", "Auto"),
    ("HDFCLIFE", "Finance"),
    ("BHARTIARTL", "Telecom"),
    ("ADANIPORTS", "Conglomerate"),
    ("LT", "Infra"),
    ("POWERGRID", "Energy"),
    ("TATASTEEL", "Metals"),
]

CASH = 50_000.0


def csv_path(symbol: str) -> str:
    return f"data/{symbol.replace('&', '')}_3min.csv"


HTF_STOCKS = {"HDFCLIFE", "POWERGRID"}
STOCK_AFTERNOON_VOL = {"HDFCLIFE": 1.25}
STOCK_DIR = {
    "ADANIPORTS": "short",
    "LT": "short",
    "TATASTEEL": "long",
    "POWERGRID": "long",
}


def run_one(df: pd.DataFrame, symbol: str = ""):
    cerebro = bt.Cerebro()
    feed = bt.feeds.PandasData(
        dataname=df, datetime=None,
        open="open", high="high", low="low", close="close",
        volume="volume", openinterest=-1,
    )
    cerebro.adddata(feed)
    if symbol in HTF_STOCKS:
        feed_htf = bt.feeds.PandasData(
            dataname=df, datetime=None,
            open="open", high="high", low="low", close="close",
            volume="volume", openinterest=-1,
        )
        cerebro.resampledata(feed_htf, timeframe=bt.TimeFrame.Minutes, compression=15)
    allowed = STOCK_DIR.get(symbol, "both")
    aftn_vol = STOCK_AFTERNOON_VOL.get(symbol, 1.0)
    cerebro.addstrategy(
        IntradayScalpingStrategy, printlog=False,
        allowed_dir=allowed, afternoon_vol_mult=aftn_vol,
    )
    cerebro.broker.setcash(CASH)
    cerebro.broker.addcommissioninfo(IndianBrokerCommission())
    cerebro.broker.set_checksubmit(False)
    cerebro.addanalyzer(bta.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bta.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bta.SharpeRatio, _name="sharpe", riskfreerate=0.065 / 252)
    results = cerebro.run()
    return cerebro, results[0]


def main():
    os.makedirs("reports", exist_ok=True)

    all_trades_rows: list[dict] = []
    summary_rows: list[dict] = []

    for symbol, sector in STOCKS:
        path = csv_path(symbol)
        if not os.path.exists(path):
            print(f"  {symbol:<12} SKIP (no data)")
            continue

        df = load_csv(path)
        print(f"  {symbol:<12} backtesting ... ", end="", flush=True)
        cerebro, strat = run_one(df, symbol)

        trades = strat._trade_log
        ta = strat.analyzers.trades.get_analysis()
        dd = strat.analyzers.drawdown.get_analysis()
        sharpe = strat.analyzers.sharpe.get_analysis()

        total = ta.get("total", {}).get("total", 0)
        won = ta.get("won", {}).get("total", 0)
        lost = ta.get("lost", {}).get("total", 0)
        win_rate = won / total * 100 if total else 0.0
        avg_win = ta.get("won", {}).get("pnl", {}).get("average", 0) if won else 0.0
        avg_loss = ta.get("lost", {}).get("pnl", {}).get("average", 0) if lost else 0.0
        max_dd = dd.get("max", {}).get("drawdown", 0)
        max_dd_money = dd.get("max", {}).get("moneydown", 0)
        sr = sharpe.get("sharperatio")
        net_pnl = cerebro.broker.getvalue() - CASH
        pf = abs(avg_win * won) / abs(avg_loss * lost) if lost and avg_loss else 0.0

        longs = [t for t in trades if t.get("direction") == "LONG"]
        shorts = [t for t in trades if t.get("direction") == "SHORT"]

        print(f"{total} trades")

        for i, t in enumerate(trades, 1):
            et = t.get("entry_time")
            xt = t.get("exit_time")
            all_trades_rows.append({
                "Symbol": symbol,
                "Sector": sector,
                "Trade#": i,
                "Call": t.get("direction", ""),
                "Signal Time": t.get("signal_time", "").strftime("%Y-%m-%d %H:%M") if t.get("signal_time") else "",
                "Entry Time": et.strftime("%Y-%m-%d %H:%M") if et else "",
                "Entry Price": round(t.get("entry_price", 0), 2),
                "Exit Time": xt.strftime("%Y-%m-%d %H:%M") if xt else "",
                "Exit Price": round(t.get("exit_price", 0), 2),
                "Exit Reason": t.get("exit_reason", ""),
                "SL Price": round(t.get("sl_price", 0), 2),
                "TP Price": round(t.get("tp_price", 0), 2),
                "Qty": t.get("qty", 0),
                "Gross P&L": round(t.get("pnl", 0), 2),
                "Net P&L": round(t.get("pnlcomm", 0), 2),
                "ATR": round(t.get("signal_atr", 0), 2),
                "RSI": round(t.get("signal_rsi", 0), 1),
                "VWAP": round(t.get("signal_vwap", 0), 2),
                "Signal Close": round(t.get("signal_close", 0), 2),
                "Duration (bars)": t.get("bars", 0),
                "Result": "WIN" if t.get("pnlcomm", 0) > 0 else "LOSS",
            })

        best = max((t.get("pnlcomm", 0) for t in trades), default=0)
        worst = min((t.get("pnlcomm", 0) for t in trades), default=0)
        long_pnl = sum(t.get("pnlcomm", 0) for t in longs)
        short_pnl = sum(t.get("pnlcomm", 0) for t in shorts)
        long_wins = sum(1 for t in longs if t.get("pnlcomm", 0) > 0)
        short_wins = sum(1 for t in shorts if t.get("pnlcomm", 0) > 0)

        summary_rows.append({
            "Symbol": symbol,
            "Sector": sector,
            "Data From": df.index[0].strftime("%Y-%m-%d"),
            "Data To": df.index[-1].strftime("%Y-%m-%d"),
            "Candles": len(df),
            "Total Trades": total,
            "Wins": won,
            "Losses": lost,
            "Win Rate %": round(win_rate, 1),
            "Net P&L": round(net_pnl, 2),
            "Avg Win": round(avg_win, 2),
            "Avg Loss": round(avg_loss, 2),
            "Best Trade": round(best, 2),
            "Worst Trade": round(worst, 2),
            "Profit Factor": round(pf, 2),
            "Max Drawdown %": round(max_dd, 2),
            "Max Drawdown INR": round(max_dd_money, 2),
            "Sharpe Ratio": round(sr, 3) if sr is not None else "",
            "Long Trades": len(longs),
            "Long Wins": long_wins,
            "Long P&L": round(long_pnl, 2),
            "Short Trades": len(shorts),
            "Short Wins": short_wins,
            "Short P&L": round(short_pnl, 2),
            "Better Side": "LONG" if long_pnl >= short_pnl else "SHORT",
            "Verdict": (
                "TRADEABLE" if pf >= 1.0 and win_rate >= 45 else
                "MARGINAL" if pf >= 0.7 else
                "AVOID"
            ),
        })

    trades_df = pd.DataFrame(all_trades_rows)
    trades_df.to_csv("reports/all_trades.csv", index=False)
    print(f"\n  -> reports/all_trades.csv       ({len(trades_df)} trades)")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("reports/stock_summary.csv", index=False)
    print(f"  -> reports/stock_summary.csv    ({len(summary_df)} stocks)")

    print(f"  -> reports/backtest_report.txt  (full console report)")
    print("\nDone.")


if __name__ == "__main__":
    main()

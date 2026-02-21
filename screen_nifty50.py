#!/usr/bin/env python3
"""
Nifty 50 Stock Screener for Intraday Scalping

For each stock, runs the strategy in two modes:
  1. Base   — no higher-timeframe filter
  2. HTF    — 15-min 21-EMA trend confirmation

Picks the more profitable config per stock, ranks all stocks,
and outputs a shortlist of tradeable candidates.

Usage:
  python screen_nifty50.py              # fetch fresh 90-day data + screen
  python screen_nifty50.py --cached     # use cached CSVs only
  python screen_nifty50.py --days 180   # screen over 180 days
"""

import argparse
import os
import sys
import time as _time
from datetime import datetime

import backtrader as bt
import backtrader.analyzers as bta
import pandas as pd

from fetch_data import AngelOneClient, load_csv, save_csv
from strategy import IndianBrokerCommission, IntradayScalpingStrategy

# ─── All Nifty 50 Constituents ────────────────────────────────────────────────

STOCKS = [
    ("ADANIENT", "Conglomerate"),
    ("ADANIPORTS", "Conglomerate"),
    ("APOLLOHOSP", "Healthcare"),
    ("ASIANPAINT", "Consumer"),
    ("AXISBANK", "Finance"),
    ("BAJAJ-AUTO", "Auto"),
    ("BAJAJFINSV", "Finance"),
    ("BAJFINANCE", "Finance"),
    ("BEL", "Defence"),
    ("BHARTIARTL", "Telecom"),
    ("CIPLA", "Pharma"),
    ("COALINDIA", "Metals"),
    ("DRREDDY", "Pharma"),
    ("EICHERMOT", "Auto"),
    ("GRASIM", "Cement"),
    ("HCLTECH", "IT"),
    ("HDFCBANK", "Finance"),
    ("HDFCLIFE", "Finance"),
    ("HINDALCO", "Metals"),
    ("HINDUNILVR", "FMCG"),
    ("ICICIBANK", "Finance"),
    ("INDIGO", "Aviation"),
    ("INFY", "IT"),
    ("ITC", "FMCG"),
    ("JIOFIN", "Finance"),
    ("JSWSTEEL", "Metals"),
    ("KOTAKBANK", "Finance"),
    ("LT", "Infra"),
    ("M&M", "Auto"),
    ("MARUTI", "Auto"),
    ("NESTLEIND", "FMCG"),
    ("NTPC", "Energy"),
    ("ONGC", "Energy"),
    ("POWERGRID", "Energy"),
    ("RELIANCE", "Conglomerate"),
    ("SBILIFE", "Finance"),
    ("SBIN", "Finance"),
    ("SHRIRAMFIN", "Finance"),
    ("SUNPHARMA", "Pharma"),
    ("TATACONSUM", "FMCG"),
    ("TATAMOTORS", "Auto"),
    ("TATASTEEL", "Metals"),
    ("TCS", "IT"),
    ("TECHM", "IT"),
    ("TITAN", "Consumer"),
    ("TRENT", "Consumer"),
    ("ULTRACEMCO", "Cement"),
    ("WIPRO", "IT"),
]

CASH = 50_000.0
W = 110


def csv_path(symbol: str) -> str:
    return f"data/{symbol.replace('&', '')}_3min.csv"


# ─── Data ─────────────────────────────────────────────────────────────────────


def fetch_all(days: int, cached_only: bool = False) -> dict[str, pd.DataFrame]:
    os.makedirs("data", exist_ok=True)
    data: dict[str, pd.DataFrame] = {}
    client = None

    for symbol, _ in STOCKS:
        path = csv_path(symbol)

        if os.path.exists(path):
            data[symbol] = load_csv(path)
            continue

        if cached_only:
            continue

        if client is None:
            client = AngelOneClient()
            client.connect()

        try:
            print(f"    {symbol:<12} fetching {days}d ...", end="", flush=True)
            df = client.fetch_history(symbol, days=days)
            save_csv(df, path)
            data[symbol] = df
            print(f" {len(df)} candles")
            _time.sleep(3)
        except Exception as e:
            print(f" FAILED: {e}")

    return data


# ─── Backtest Engine ──────────────────────────────────────────────────────────


def run_one(df: pd.DataFrame, use_htf: bool = False):
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
    cerebro.addstrategy(
        IntradayScalpingStrategy, printlog=False,
        use_htf_filter=use_htf,
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


def extract_metrics(cerebro, strat) -> dict:
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
    max_dd_inr = dd.get("max", {}).get("moneydown", 0)
    sr = sharpe.get("sharperatio")
    net_pnl = cerebro.broker.getvalue() - CASH
    pf = abs(avg_win * won) / abs(avg_loss * lost) if lost and avg_loss else 0.0

    trades = strat._trade_log
    longs = [t for t in trades if t.get("direction") == "LONG"]
    shorts = [t for t in trades if t.get("direction") == "SHORT"]
    long_wins = sum(1 for t in longs if t.get("pnlcomm", 0) > 0)
    short_wins = sum(1 for t in shorts if t.get("pnlcomm", 0) > 0)
    long_pnl = sum(t.get("pnlcomm", 0) for t in longs)
    short_pnl = sum(t.get("pnlcomm", 0) for t in shorts)

    if pf >= 1.0 and win_rate >= 45:
        verdict = "TRADEABLE"
    elif pf >= 0.7:
        verdict = "MARGINAL"
    else:
        verdict = "AVOID"

    return {
        "total": total, "won": won, "lost": lost,
        "win_rate": round(win_rate, 1),
        "net_pnl": round(net_pnl, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "pf": round(pf, 2),
        "max_dd": round(max_dd, 2),
        "max_dd_inr": round(max_dd_inr, 2),
        "sharpe": round(sr, 3) if sr is not None else None,
        "longs": len(longs), "long_wins": long_wins, "long_pnl": round(long_pnl, 2),
        "shorts": len(shorts), "short_wins": short_wins, "short_pnl": round(short_pnl, 2),
        "best_side": "LONG" if long_pnl >= short_pnl else "SHORT",
        "verdict": verdict,
    }


# ─── Screening ────────────────────────────────────────────────────────────────


def screen_stock(df: pd.DataFrame) -> tuple[dict, dict]:
    c_base, strat_base = run_one(df, use_htf=False)
    m_base = extract_metrics(c_base, strat_base)

    c_htf, strat_htf = run_one(df, use_htf=True)
    m_htf = extract_metrics(c_htf, strat_htf)

    return m_base, m_htf


def main():
    parser = argparse.ArgumentParser(description="Nifty 50 intraday scalping screener")
    parser.add_argument("--cached", action="store_true", help="Only use cached CSVs")
    parser.add_argument("--days", type=int, default=90, help="Lookback period (default: 90)")
    args = parser.parse_args()

    run_date = datetime.now().strftime("%Y-%m-%d %H:%M")

    print(f"\n{'=' * W}")
    print(f"  NIFTY 50 INTRADAY SCALPING SCREENER")
    print(f"  Run: {run_date}  |  Lookback: {args.days}d  |  Capital: {CASH:,.0f}")
    print(f"  Strategy: EMA(9/21) + VWAP + RSI + Volume  |  RR 1:1.5  |  Slippage 0.01%")
    print(f"  Testing each stock: BASE (no filter) vs HTF (15-min 21-EMA)")
    print(f"{'=' * W}")

    # ── Fetch ──
    print(f"\n  [1/3] LOADING DATA ...\n")
    all_data = fetch_all(args.days, cached_only=args.cached)
    loaded = len(all_data)
    print(f"\n    {loaded}/{len(STOCKS)} stocks loaded\n")

    if loaded == 0:
        print("  No data available. Run without --cached to fetch from Angel One.")
        sys.exit(1)

    # ── Screen ──
    print(f"  [2/3] SCREENING (2 backtests per stock) ...\n")
    print(f"    {'Stock':<12} {'Base P&L':>10}  {'HTF P&L':>10}  {'Pick':>5}  {'Trades':>7}  {'Win%':>6}  {'PF':>5}  {'MaxDD':>6}")
    print(f"    {'-' * 72}")

    rows: list[dict] = []

    for symbol, sector in STOCKS:
        if symbol not in all_data:
            continue

        df = all_data[symbol]
        m_base, m_htf = screen_stock(df)

        htf_better = m_htf["net_pnl"] > m_base["net_pnl"]
        best = m_htf if htf_better else m_base
        pick = "HTF" if htf_better else "BASE"

        rows.append({
            "Symbol": symbol,
            "Sector": sector,
            "HTF Recommended": pick,
            "Base P&L": m_base["net_pnl"],
            "Base Trades": m_base["total"],
            "Base Win%": m_base["win_rate"],
            "Base PF": m_base["pf"],
            "Base MaxDD%": m_base["max_dd"],
            "HTF P&L": m_htf["net_pnl"],
            "HTF Trades": m_htf["total"],
            "HTF Win%": m_htf["win_rate"],
            "HTF PF": m_htf["pf"],
            "HTF MaxDD%": m_htf["max_dd"],
            "Best P&L": best["net_pnl"],
            "Best Trades": best["total"],
            "Best Win%": best["win_rate"],
            "Best PF": best["pf"],
            "Best MaxDD%": best["max_dd"],
            "Best MaxDD INR": best["max_dd_inr"],
            "Sharpe": best["sharpe"],
            "Longs": best["longs"],
            "Long Wins": best["long_wins"],
            "Long P&L": best["long_pnl"],
            "Shorts": best["shorts"],
            "Short Wins": best["short_wins"],
            "Short P&L": best["short_pnl"],
            "Best Side": best["best_side"],
            "Verdict": best["verdict"],
        })

        tag = "+" if best["net_pnl"] > 0 else " "
        print(
            f"    {symbol:<12} {m_base['net_pnl']:>+10,.2f}  {m_htf['net_pnl']:>+10,.2f}"
            f"  {pick:>5}  {best['total']:>7}  {best['win_rate']:>5.1f}%  {best['pf']:>5.2f}  {best['max_dd']:>5.2f}%"
        )

    # ── Sort & Report ──
    rows.sort(key=lambda r: r["Best P&L"], reverse=True)

    print(f"\n  [3/3] RESULTS (Ranked by Best P&L)\n")
    print(f"  {'=' * (W - 4)}")
    print(
        f"  {'Rank':>4}  {'Symbol':<12} {'Sector':<14} {'Config':>6} {'Trades':>7} "
        f"{'Win%':>6} {'Net P&L':>12} {'MaxDD%':>7} {'PF':>6} {'Side':<6} {'Verdict':<10}"
    )
    print(f"  {'-' * (W - 4)}")

    tradeable: list[dict] = []
    total_best_pnl = 0.0

    for i, r in enumerate(rows, 1):
        pnl_s = f"{r['Best P&L']:>+12,.2f}"
        print(
            f"  {i:>4}  {r['Symbol']:<12} {r['Sector']:<14} {r['HTF Recommended']:>6} "
            f"{r['Best Trades']:>7} {r['Best Win%']:>5.1f}% {pnl_s} "
            f"{r['Best MaxDD%']:>6.2f}% {r['Best PF']:>6.2f} {r['Best Side']:<6} {r['Verdict']:<10}"
        )
        total_best_pnl += r["Best P&L"]
        if r["Verdict"] == "TRADEABLE":
            tradeable.append(r)

    print(f"  {'-' * (W - 4)}")
    print(f"  {'':>4}  {'TOTAL':<12} {'':>14} {'':>6} {sum(r['Best Trades'] for r in rows):>7} "
          f"{'':>6} {total_best_pnl:>+12,.2f}")

    # ── Shortlist ──
    if tradeable:
        print(f"\n  {'=' * (W - 4)}")
        print(f"  RECOMMENDED SHORTLIST ({len(tradeable)} stocks)")
        print(f"  {'=' * (W - 4)}")
        print(
            f"  {'Symbol':<12} {'Sector':<14} {'HTF':>4} {'Trades':>7} {'Win%':>6} "
            f"{'Net P&L':>12} {'MaxDD%':>7} {'PF':>6} {'Longs':>6} {'Shorts':>7} {'Side':<6}"
        )
        print(f"  {'-' * (W - 4)}")
        shortlist_pnl = 0.0
        for r in tradeable:
            pnl_s = f"{r['Best P&L']:>+12,.2f}"
            htf_tag = "Yes" if r["HTF Recommended"] == "HTF" else "No"
            print(
                f"  {r['Symbol']:<12} {r['Sector']:<14} {htf_tag:>4} "
                f"{r['Best Trades']:>7} {r['Best Win%']:>5.1f}% {pnl_s} "
                f"{r['Best MaxDD%']:>6.2f}% {r['Best PF']:>6.2f} "
                f"{r['Longs']:>6} {r['Shorts']:>7} {r['Best Side']:<6}"
            )
            shortlist_pnl += r["Best P&L"]
        print(f"  {'-' * (W - 4)}")
        print(f"  SHORTLIST PORTFOLIO P&L: {shortlist_pnl:>+12,.2f}")
        print(f"  {'=' * (W - 4)}")

    # ── Save CSV ──
    os.makedirs("reports", exist_ok=True)
    df_out = pd.DataFrame(rows)
    out_path = "reports/nifty50_screen.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path} ({len(rows)} stocks)")

    # ── Config snippet for batch_backtest.py ──
    if tradeable:
        htf_set = [r["Symbol"] for r in tradeable if r["HTF Recommended"] == "HTF"]
        print("\n  ── Copy into batch_backtest.py / generate_report.py ──")
        print("  STOCKS = [")
        for r in tradeable:
            print(f'      ("{r["Symbol"]}", "{r["Sector"]}"),')
        print("  ]")
        htf_items = ", ".join(f'"{s}"' for s in htf_set)
        print(f"  HTF_STOCKS = {{{htf_items}}}")

    print()


if __name__ == "__main__":
    main()

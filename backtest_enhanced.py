#!/usr/bin/env python3
"""
Backtest runner for Enhanced Strategy improvements.

Tests 9 variants (+ optional index alignment) against the baseline and
produces a side-by-side comparison with delta P&L per variant.

Variants:
  baseline      — current strategy (fixed SL/TP bracket)
  rsi_tight     — RSI LONG 50-75, SHORT 25-50
  vwap_ext      — reject signals > 1.5×ATR from VWAP
  vol_cap       — reject volume > 3.5× SMA (exhaustion climax)
  gap_filter    — suppress signals on ≥1.5% gap mornings
  pdhl          — respect PDH/PDL as support/resistance
  all_filters   — all 5 filters combined
  scale_out     — 50% TP at 1R → SL to breakeven → TP2 at 2.5R
  scale_combo   — scale-out + all filters
  index_align   — (optional) align with Nifty 50 VWAP

Usage:
  python backtest_enhanced.py                         # all 9 stocks
  python backtest_enhanced.py --stocks TATASTEEL TITAN
"""

import argparse
import csv
import os
import sys

import backtrader as bt
import backtrader.analyzers as bta
import pandas as pd

from fetch_data import load_csv
from strategy import IndianBrokerCommission, IntradayScalpingStrategy
from strategy_enhanced import EnhancedStrategy

# ─── Configuration ────────────────────────────────────────────────────────────

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

CASH = 50_000.0

NIFTY_CSV = "data/NIFTY50_3min.csv"

VARIANTS = [
    ("baseline", IntradayScalpingStrategy, {}),
    ("rsi_tight", EnhancedStrategy, {
        "long_rsi_low": 50, "long_rsi_high": 75,
        "short_rsi_low": 25, "short_rsi_high": 50,
    }),
    ("vwap_ext", EnhancedStrategy, {"vwap_ext_atr": 1.5}),
    ("vol_cap", EnhancedStrategy, {"vol_cap_mult": 3.5}),
    ("gap_filter", EnhancedStrategy, {"gap_threshold": 0.015}),
    ("pdhl", EnhancedStrategy, {"pdhl_buffer_atr": 0.2}),
    ("all_filters", EnhancedStrategy, {
        "long_rsi_low": 50, "long_rsi_high": 75,
        "short_rsi_low": 25, "short_rsi_high": 50,
        "vwap_ext_atr": 1.5,
        "vol_cap_mult": 3.5,
        "gap_threshold": 0.015,
        "pdhl_buffer_atr": 0.2,
    }),
    ("scale_out", EnhancedStrategy, {"use_scale_out": True}),
    ("scale_combo", EnhancedStrategy, {
        "use_scale_out": True,
        "long_rsi_low": 50, "long_rsi_high": 75,
        "short_rsi_low": 25, "short_rsi_high": 50,
        "vwap_ext_atr": 1.5,
        "vol_cap_mult": 3.5,
        "gap_threshold": 0.015,
        "pdhl_buffer_atr": 0.2,
    }),
]


def csv_path(symbol: str) -> str:
    return f"data/{symbol.replace('&', '')}_3min.csv"


# ─── Data ─────────────────────────────────────────────────────────────────────


def load_all(
    filter_symbols: list[str] | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame | None]:
    data: dict[str, pd.DataFrame] = {}
    for symbol, _ in STOCKS:
        if filter_symbols and symbol not in filter_symbols:
            continue
        path = csv_path(symbol)
        if not os.path.exists(path):
            print(f"  {symbol:<12}  SKIP — no cached CSV")
            continue
        data[symbol] = load_csv(path)
        print(f"  {symbol:<12}  {len(data[symbol]):>5} candles")

    nifty_df = None
    if os.path.exists(NIFTY_CSV):
        nifty_df = load_csv(NIFTY_CSV)
        print(f"  {'NIFTY50':<12}  {len(nifty_df):>5} candles  (index)")

    return data, nifty_df


# ─── Backtest Engine ──────────────────────────────────────────────────────────


def run_variant(
    df: pd.DataFrame,
    symbol: str,
    strat_cls,
    strat_kwargs: dict,
    nifty_df: pd.DataFrame | None = None,
) -> dict:
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

    if nifty_df is not None and strat_kwargs.get("use_index_filter"):
        idx_kw = dict(
            dataname=nifty_df, datetime=None,
            open="open", high="high", low="low", close="close",
            volume="volume", openinterest=-1,
        )
        cerebro.adddata(bt.feeds.PandasData(**idx_kw))

    use_htf = symbol in HTF_STOCKS
    cerebro.addstrategy(
        strat_cls,
        printlog=False,
        use_htf_filter=use_htf,
        **strat_kwargs,
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
    strat = results[0]
    return _extract_metrics(cerebro, strat)


def _extract_metrics(cerebro, strat) -> dict:
    ta = strat.analyzers.trades.get_analysis()
    dd = strat.analyzers.drawdown.get_analysis()
    sharpe = strat.analyzers.sharpe.get_analysis()

    total = ta.get("total", {}).get("total", 0)
    won = ta.get("won", {}).get("total", 0)
    lost = ta.get("lost", {}).get("total", 0)
    avg_win = ta.get("won", {}).get("pnl", {}).get("average", 0) if won else 0
    avg_loss = ta.get("lost", {}).get("pnl", {}).get("average", 0) if lost else 0

    win_rate = won / total * 100 if total else 0.0
    pf = abs(avg_win * won) / abs(avg_loss * lost) if lost and avg_loss else 0.0
    net_pnl = cerebro.broker.getvalue() - CASH
    max_dd = dd.get("max", {}).get("drawdown", 0)
    sr = sharpe.get("sharperatio")

    tlog = strat._trade_log
    exit_counts: dict[str, int] = {}
    tp1_hits = 0
    for t in tlog:
        r = t.get("exit_reason", "UNKNOWN")
        exit_counts[r] = exit_counts.get(r, 0) + 1
        if t.get("tp1_hit"):
            tp1_hits += 1

    return {
        "total": total,
        "won": won,
        "lost": lost,
        "win_rate": win_rate,
        "net_pnl": net_pnl,
        "pf": pf,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "max_dd": max_dd,
        "sharpe": sr,
        "exits": exit_counts,
        "tp1_hits": tp1_hits,
    }


# ─── Reporting ────────────────────────────────────────────────────────────────

W = 145


def _pnl(v: float) -> str:
    return f"{'+' if v > 0 else ''}{v:,.2f}"


def _variant_list():
    return VARIANTS


def print_stock_table(symbol: str, results: dict[str, dict]) -> None:
    base = results.get("baseline", {})

    print(f"\n{'=' * W}")
    print(f"  {symbol}")
    print(f"{'=' * W}")

    hdr = (
        f"  {'Variant':<14} {'Trades':>6} {'Won':>4} {'Win%':>6} "
        f"{'Net P&L':>11} {'ΔP&L':>10} {'PF':>5} {'AvgWin':>9} {'AvgLoss':>9} "
        f"{'MaxDD%':>7} {'Sharpe':>7} "
        f"{'SL':>4} {'TP':>4} {'TP2':>4} {'BE':>3} {'SQR':>4}"
    )
    print(hdr)
    print(f"  {'-' * (W - 4)}")

    for name, _, _ in _variant_list():
        m = results.get(name)
        if m is None:
            continue

        delta = m["net_pnl"] - base.get("net_pnl", 0) if base else 0
        sr_s = f"{m['sharpe']:.3f}" if m["sharpe"] is not None else "  N/A"
        ex = m["exits"]

        print(
            f"  {name:<14} {m['total']:>6} {m['won']:>4} {m['win_rate']:>5.1f}% "
            f"{_pnl(m['net_pnl']):>11} {_pnl(delta):>10} {m['pf']:>5.2f} "
            f"{_pnl(m['avg_win']):>9} {_pnl(m['avg_loss']):>9} "
            f"{m['max_dd']:>6.2f}% {sr_s:>7} "
            f"{ex.get('SL', 0):>4} {ex.get('TP', 0):>4} "
            f"{ex.get('TP2', 0):>4} {ex.get('SL_BREAKEVEN', 0):>3} "
            f"{ex.get('SQUAREOFF', 0):>4}"
        )


def print_aggregate(all_results: dict[str, dict[str, dict]]) -> None:
    print(f"\n{'=' * W}")
    print(f"  AGGREGATE COMPARISON — ALL STOCKS")
    print(f"{'=' * W}")

    hdr = (
        f"  {'Variant':<14} {'Trades':>6} {'Won':>4} {'Win%':>6} "
        f"{'Net P&L':>11} {'ΔP&L':>10} {'PF':>5} {'AvgWin':>9} {'AvgLoss':>9} "
        f"{'MaxDD%':>7} "
        f"{'SL':>4} {'TP':>4} {'TP2':>4} {'BE':>3} {'SQR':>4}"
    )
    print(hdr)
    print(f"  {'-' * (W - 4)}")

    base_agg_pnl = 0.0
    for sym in all_results:
        base_agg_pnl += all_results[sym].get("baseline", {}).get("net_pnl", 0)

    for name, _, _ in _variant_list():
        totals = {
            "total": 0, "won": 0, "lost": 0, "net_pnl": 0.0,
            "sum_avg_win": 0.0, "sum_avg_loss": 0.0, "max_dd": 0.0,
            "cnt_win": 0, "cnt_loss": 0,
        }
        exits_agg: dict[str, int] = {}
        stock_count = 0

        for sym, stock_res in all_results.items():
            m = stock_res.get(name)
            if m is None:
                continue
            stock_count += 1
            totals["total"] += m["total"]
            totals["won"] += m["won"]
            totals["lost"] += m["lost"]
            totals["net_pnl"] += m["net_pnl"]
            if m["won"]:
                totals["sum_avg_win"] += m["avg_win"]
                totals["cnt_win"] += 1
            if m["lost"]:
                totals["sum_avg_loss"] += m["avg_loss"]
                totals["cnt_loss"] += 1
            totals["max_dd"] = max(totals["max_dd"], m["max_dd"])
            for k, v in m["exits"].items():
                exits_agg[k] = exits_agg.get(k, 0) + v

        if stock_count == 0:
            continue

        t = totals["total"]
        wr = totals["won"] / t * 100 if t else 0
        aw = (
            totals["sum_avg_win"] / totals["cnt_win"]
            if totals["cnt_win"] else 0
        )
        al = (
            totals["sum_avg_loss"] / totals["cnt_loss"]
            if totals["cnt_loss"] else 0
        )
        gross_win = abs(aw * totals["won"])
        gross_loss = abs(al * totals["lost"])
        pf = gross_win / gross_loss if gross_loss else 0
        delta = totals["net_pnl"] - base_agg_pnl

        print(
            f"  {name:<14} {totals['total']:>6} {totals['won']:>4} {wr:>5.1f}% "
            f"{_pnl(totals['net_pnl']):>11} {_pnl(delta):>10} {pf:>5.2f} "
            f"{_pnl(aw):>9} {_pnl(al):>9} "
            f"{totals['max_dd']:>6.2f}% "
            f"{exits_agg.get('SL', 0):>4} {exits_agg.get('TP', 0):>4} "
            f"{exits_agg.get('TP2', 0):>4} {exits_agg.get('SL_BREAKEVEN', 0):>3} "
            f"{exits_agg.get('SQUAREOFF', 0):>4}"
        )

    print(f"{'=' * W}")


def export_csv(all_results: dict[str, dict[str, dict]]) -> str:
    os.makedirs("reports", exist_ok=True)
    path = "reports/enhanced_comparison.csv"

    rows = []
    for sym in all_results:
        for name, _, _ in _variant_list():
            m = all_results[sym].get(name)
            if m is None:
                continue
            ex = m["exits"]
            rows.append({
                "symbol": sym,
                "variant": name,
                "total_trades": m["total"],
                "won": m["won"],
                "lost": m["lost"],
                "win_rate": round(m["win_rate"], 2),
                "net_pnl": round(m["net_pnl"], 2),
                "profit_factor": round(m["pf"], 2),
                "avg_win": round(m["avg_win"], 2),
                "avg_loss": round(m["avg_loss"], 2),
                "max_drawdown_pct": round(m["max_dd"], 2),
                "sharpe": (
                    round(m["sharpe"], 3) if m["sharpe"] is not None else ""
                ),
                "exits_sl": ex.get("SL", 0),
                "exits_tp": ex.get("TP", 0),
                "exits_tp2": ex.get("TP2", 0),
                "exits_sl_breakeven": ex.get("SL_BREAKEVEN", 0),
                "exits_squareoff": ex.get("SQUAREOFF", 0),
                "tp1_hits": m.get("tp1_hits", 0),
            })

    fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return path


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Backtest enhanced strategy improvements",
    )
    parser.add_argument(
        "--stocks", nargs="+", default=None,
        help="Run only these symbols (default: all 9)",
    )
    args = parser.parse_args()

    # Conditionally add index alignment variant
    nifty_available = os.path.exists(NIFTY_CSV)
    if nifty_available:
        VARIANTS.append((
            "index_align", EnhancedStrategy, {"use_index_filter": True},
        ))

    n_variants = len(VARIANTS)

    print(f"\n{'=' * W}")
    print(
        f"  ENHANCED STRATEGY BACKTEST COMPARISON\n"
        f"  Capital: ₹{CASH:,.0f}  |  {n_variants} variants  |  "
        f"{'all' if args.stocks is None else len(args.stocks)} stocks"
    )
    if not nifty_available:
        print(
            f"  Index alignment: SKIPPED (no {NIFTY_CSV})"
        )
    print(f"{'=' * W}")

    # ── Load data ──
    print(f"\n[1/3] Loading cached data ...\n")
    stock_data, nifty_df = load_all(args.stocks)

    if not stock_data:
        print("No cached data found. Run batch_backtest.py first to fetch data.")
        sys.exit(1)

    print(f"\n  -> {len(stock_data)} stocks loaded\n")

    # ── Backtest ──
    print(
        f"[2/3] Running {n_variants} variants × "
        f"{len(stock_data)} stocks ...\n"
    )

    all_results: dict[str, dict[str, dict]] = {}

    for symbol in stock_data:
        df = stock_data[symbol]
        all_results[symbol] = {}
        print(f"  {symbol:<12}", end="", flush=True)

        for name, strat_cls, kwargs in VARIANTS:
            try:
                metrics = run_variant(
                    df, symbol, strat_cls, kwargs, nifty_df=nifty_df,
                )
                all_results[symbol][name] = metrics
                print(".", end="", flush=True)
            except Exception as e:
                print(f"\n    {name} FAILED: {e}")

        base_t = all_results[symbol].get("baseline", {}).get("total", 0)
        print(f"  done ({base_t} base trades)")

    # ── Reports ──
    print(f"\n[3/3] Results\n")

    for symbol in all_results:
        print_stock_table(symbol, all_results[symbol])

    print_aggregate(all_results)

    csv_out = export_csv(all_results)
    print(f"\n  CSV exported → {csv_out}\n")


if __name__ == "__main__":
    main()

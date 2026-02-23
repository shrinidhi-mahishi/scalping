#!/usr/bin/env python3
"""
Verify whether vol_cap (3.5× volume ceiling) improves P&L
on top of the proven HTF + Stagnation live configuration.

Baseline: the exact per-stock config from live_signals.py (+25,369)
Test:     same config + vol_cap_mult=3.5

Creates a combined strategy (stagnation exit + vol_cap) since
EarlyExitStrategy and EnhancedStrategy are separate class trees.
"""

import os
import sys

import backtrader as bt
import backtrader.analyzers as bta
import pandas as pd

from fetch_data import load_csv
from strategy import IndianBrokerCommission, IntradayScalpingStrategy
from strategy_early_exit import EarlyExitStrategy
from strategy_enhanced import EnhancedStrategy

# ─── Configuration (mirrors live_signals.py) ──────────────────────────────────

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


# ─── Combined Strategy: Stagnation Exit + Volume Cap ──────────────────────────


class StagnationVolCapStrategy(EarlyExitStrategy):
    """EarlyExitStrategy (stagnation) with an added volume climax cap."""

    params = (
        ("vol_cap_mult", 0),
    )

    def next(self):
        self._update_regime()

        if (
            self.position
            and not self._order_pending
            and not self._is_squareoff()
            and self._sl_order is not None
        ):
            if self._check_early_exit():
                reason = self.p.exit_mode.upper()
                self._cancel_bracket()
                if self._active_trade is not None:
                    self._active_trade["exit_reason"] = reason
                self.close()
                self._log(
                    f"EARLY EXIT ({reason}) @ ₹{self.data.close[0]:.2f}"
                )
                return

        if self._is_squareoff():
            if self.position:
                self._cancel_bracket()
                if self._active_trade is not None:
                    self._active_trade["exit_reason"] = "SQUAREOFF"
                self.close()
                self._log(f"SQUAREOFF @ ₹{self.data.close[0]:.2f}")
            return

        if self._order_pending:
            return
        if self.position:
            return
        if not self._in_entry_window():
            return

        close = self.data.close[0]
        vwap = self.vwap[0]
        rsi_val = self.rsi[0]
        vol = self.data.volume[0]
        vol_avg = self.vol_sma[0]
        cross = self.ema_cross[0]
        atr_val = self.atr[0]

        if self.p.use_regime_filter and self._regime_ema_val is not None:
            allow_long = close > self._regime_ema_val
            allow_short = close < self._regime_ema_val
        else:
            allow_long = True
            allow_short = True

        if self.p.use_htf_filter and hasattr(self, "htf_ema"):
            allow_long = allow_long and close > self.htf_ema[0]
            allow_short = allow_short and close < self.htf_ema[0]

        vol_thresh = vol_avg * self.p.vol_mult
        vol_ok = vol > vol_thresh
        if self.p.vol_cap_mult > 0 and vol_avg > 0:
            vol_ok = vol_ok and vol < vol_avg * self.p.vol_cap_mult

        if (
            allow_long
            and close > vwap
            and cross > 0
            and vol_ok
            and self.p.long_rsi_low <= rsi_val <= self.p.long_rsi_high
        ):
            qty = self._calc_qty(atr_val)
            self._current_qty = qty
            self._pending_dir = "long"
            self._signal_atr = atr_val
            self._order_pending = True
            self._active_trade = {
                "direction": "LONG",
                "signal_time": self.data.datetime.datetime(0),
                "signal_close": close,
                "signal_vwap": vwap,
                "signal_rsi": rsi_val,
                "signal_atr": atr_val,
            }
            order = self.buy(size=qty)
            self._main_ref = order.ref
            self._log(
                f"LONG SIGNAL   Close={close:.2f}  VWAP={vwap:.2f}  "
                f"RSI={rsi_val:.1f}  ATR={atr_val:.2f}  Qty={qty}"
            )

        elif (
            allow_short
            and close < vwap
            and cross < 0
            and vol_ok
            and self.p.short_rsi_low <= rsi_val <= self.p.short_rsi_high
        ):
            qty = self._calc_qty(atr_val)
            self._current_qty = qty
            self._pending_dir = "short"
            self._signal_atr = atr_val
            self._order_pending = True
            self._active_trade = {
                "direction": "SHORT",
                "signal_time": self.data.datetime.datetime(0),
                "signal_close": close,
                "signal_vwap": vwap,
                "signal_rsi": rsi_val,
                "signal_atr": atr_val,
            }
            order = self.sell(size=qty)
            self._main_ref = order.ref
            self._log(
                f"SHORT SIGNAL  Close={close:.2f}  VWAP={vwap:.2f}  "
                f"RSI={rsi_val:.1f}  ATR={atr_val:.2f}  Qty={qty}"
            )


# ─── Runner ───────────────────────────────────────────────────────────────────


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
    lost = ta.get("lost", {}).get("total", 0)
    avg_win = ta.get("won", {}).get("pnl", {}).get("average", 0) if won else 0
    avg_loss = ta.get("lost", {}).get("pnl", {}).get("average", 0) if lost else 0
    net_pnl = cerebro.broker.getvalue() - CASH
    wr = won / total * 100 if total else 0
    pf = abs(avg_win * won) / abs(avg_loss * lost) if lost and avg_loss else 0
    max_dd = dd.get("max", {}).get("drawdown", 0)

    return {
        "total": total, "won": won, "wr": wr,
        "net_pnl": net_pnl, "pf": pf, "max_dd": max_dd,
    }


def get_configs(sym):
    """Return (baseline_cls, baseline_kw, volcap_cls, volcap_kw, htf, label)."""
    has_htf = sym in HTF_STOCKS
    has_stag = sym not in STAGNATION_SKIP

    stag_kw = {
        "exit_mode": "stagnation",
        "max_trade_bars": 8,
        "min_profit_atr": 0.2,
    }

    if has_stag:
        base_cls = EarlyExitStrategy
        base_kw = dict(stag_kw)
        vc_cls = StagnationVolCapStrategy
        vc_kw = {**stag_kw, "vol_cap_mult": 3.5}
        label = "HTF+S" if has_htf else "BASE+S"
    else:
        base_cls = IntradayScalpingStrategy
        base_kw = {}
        vc_cls = EnhancedStrategy
        vc_kw = {"vol_cap_mult": 3.5}
        label = "HTF" if has_htf else "BASE"

    return base_cls, base_kw, vc_cls, vc_kw, has_htf, label


def main():
    W = 130
    print(f"\n{'=' * W}")
    print(f"  VOL_CAP VERIFICATION: Does 3.5× volume cap improve the proven +25,369 baseline?")
    print(f"  Testing each stock's live config vs the same config + vol_cap_mult=3.5")
    print(f"{'=' * W}\n")

    print(
        f"  {'Stock':<12} {'Config':>6}  "
        f"{'Live P&L':>10} {'+ VolCap':>10} {'Delta':>10}  "
        f"{'Trades':>7} {'→':>1} {'Trades':>7}  "
        f"{'Win%':>6} {'→':>1} {'Win%':>6}  "
        f"{'PF':>5} {'→':>1} {'PF':>5}  "
        f"{'Verdict':>8}"
    )
    print(f"  {'-' * (W - 4)}")

    total_live = 0.0
    total_vc = 0.0
    improvements = 0
    regressions = 0

    for sym, sec in STOCKS:
        path = csv_path(sym)
        if not os.path.exists(path):
            print(f"  {sym:<12}  SKIP — no data")
            continue

        df = load_csv(path)
        base_cls, base_kw, vc_cls, vc_kw, has_htf, label = get_configs(sym)

        live = run_one(df, has_htf, base_cls, base_kw)
        volcap = run_one(df, has_htf, vc_cls, vc_kw)

        delta = volcap["net_pnl"] - live["net_pnl"]
        total_live += live["net_pnl"]
        total_vc += volcap["net_pnl"]

        if delta > 0:
            verdict = "BETTER"
            improvements += 1
        elif delta < 0:
            verdict = "WORSE"
            regressions += 1
        else:
            verdict = "SAME"

        flag = " <<" if delta < -100 else ""

        print(
            f"  {sym:<12} {label:>6}  "
            f"{live['net_pnl']:>+10,.2f} {volcap['net_pnl']:>+10,.2f} {delta:>+10,.2f}  "
            f"{live['total']:>7} {'→':>1} {volcap['total']:>7}  "
            f"{live['wr']:>5.1f}% {'→':>1} {volcap['wr']:>5.1f}%  "
            f"{live['pf']:>5.2f} {'→':>1} {volcap['pf']:>5.2f}  "
            f"{verdict:>8}{flag}"
        )

    print(f"  {'-' * (W - 4)}")
    total_delta = total_vc - total_live
    print(
        f"  {'TOTAL':<12} {'':>6}  "
        f"{total_live:>+10,.2f} {total_vc:>+10,.2f} {total_delta:>+10,.2f}  "
        f"{'':>17}  {'':>14}  {'':>12}  "
    )

    print(f"\n  Summary: {improvements} improved, {regressions} regressed, "
          f"{9 - improvements - regressions} unchanged")

    if total_delta > 0:
        print(f"\n  ✓ VOL_CAP IMPROVES the live baseline by {total_delta:+,.2f}")
        print(f"    Live baseline:      ₹{total_live:>+10,.2f}")
        print(f"    With vol_cap 3.5×:  ₹{total_vc:>+10,.2f}")
    else:
        print(f"\n  ✗ VOL_CAP HURTS the live baseline by {total_delta:+,.2f}")
        print(f"    DO NOT ADD vol_cap to live_signals.py")

    print()


if __name__ == "__main__":
    main()

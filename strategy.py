"""
Intraday Scalping Strategy for NSE 3-Minute Candles (v3 — Regime-Aware)

Indicators : Session VWAP, EMA(9/21), RSI(9), VolSMA(20), ATR(14)
Regime gate : Daily 21-EMA determines allowed trade direction per bar
HTF filter  : Optional 15-min 21-EMA trend confirmation
Entry window: 10:00-12:00  and  14:00-15:00  (midday dead zone skipped)
Squareoff   : 15:15 (MIS cutoff)
Risk-Reward : 1 : 1.5 via ATR-based bracket (SL = 1×ATR, TP = 1.5×ATR)
Sizing      : qty = (Capital × risk_pct) / ATR  (1% risk per trade)
Slippage    : 0.01% on stop/market orders; limit (TP) fills at exact price
"""

from datetime import time

import backtrader as bt


# ─── Custom VWAP Indicator ────────────────────────────────────────────────────


class SessionVWAP(bt.Indicator):
    """
    Volume-Weighted Average Price that resets at each new trading session.
    VWAP = Σ(TypicalPrice × Volume) / Σ(Volume), cumulative from session open.
    """

    lines = ("vwap",)
    plotinfo = dict(subplot=False)

    def __init__(self):
        self.addminperiod(1)
        self._cum_vol = 0.0
        self._cum_tp_vol = 0.0
        self._session_date = None

    def next(self):
        current_date = self.data.datetime.date(0)

        if current_date != self._session_date:
            self._cum_vol = 0.0
            self._cum_tp_vol = 0.0
            self._session_date = current_date

        tp = (self.data.high[0] + self.data.low[0] + self.data.close[0]) / 3.0
        self._cum_vol += self.data.volume[0]
        self._cum_tp_vol += tp * self.data.volume[0]

        if self._cum_vol > 0:
            self.lines.vwap[0] = self._cum_tp_vol / self._cum_vol
        else:
            self.lines.vwap[0] = self.data.close[0]


# ─── Indian Broker Commission (₹20 cap per order) ────────────────────────────


class IndianBrokerCommission(bt.CommInfoBase):
    """
    Realistic Indian discount broker commission with 5x intraday margin:
      per leg  = min(trade_value × 0.03%, ₹20)
      margin   = 20% of price (5x MIS leverage, matching Angel One / Zerodha)
    stocklike=False + automargin enables leveraged buying AND selling.
    mult=1.0 keeps P&L = (exit - entry) × qty, same as stocks.
    """

    params = (
        ("commission", 0.0003),
        ("max_per_order", 20.0),
        ("stocklike", False),
        ("commtype", bt.CommInfoBase.COMM_PERC),
        ("automargin", 0.20),
        ("mult", 1.0),
    )

    def _getcommission(self, size, price, pseudoexec):
        value = abs(size) * price
        comm = value * self.p.commission
        return min(comm, self.p.max_per_order)


# ─── Strategy ─────────────────────────────────────────────────────────────────


class IntradayScalpingStrategy(bt.Strategy):
    params = (
        ("fast_ema_period", 9),
        ("slow_ema_period", 21),
        ("rsi_period", 9),
        ("vol_sma_period", 20),
        ("atr_period", 14),
        ("rr_multiplier", 1.5),
        ("risk_pct", 0.01),
        ("leverage_cap", 5.0),
        ("trade_qty", 0),
        ("htf_ema_period", 21),
        ("use_htf_filter", False),
        ("use_regime_filter", False),
        ("regime_ema_period", 21),
        ("vol_mult", 1.0),
        ("long_rsi_low", 40),
        ("long_rsi_high", 70),
        ("short_rsi_low", 30),
        ("short_rsi_high", 60),
        ("printlog", True),
    )

    MORNING_OPEN = time(10, 0)
    MORNING_CLOSE = time(12, 0)
    AFTERNOON_OPEN = time(14, 0)
    AFTERNOON_CLOSE = time(15, 0)
    SQUAREOFF = time(15, 15)

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def __init__(self):
        self.vwap = SessionVWAP(self.data)
        self.fast_ema = bt.indicators.EMA(
            self.data.close, period=self.p.fast_ema_period
        )
        self.slow_ema = bt.indicators.EMA(
            self.data.close, period=self.p.slow_ema_period
        )
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        self.vol_sma = bt.indicators.SMA(
            self.data.volume, period=self.p.vol_sma_period
        )
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        self.ema_cross = bt.indicators.CrossOver(self.fast_ema, self.slow_ema)

        if self.p.use_htf_filter and len(self.datas) >= 2:
            self.htf_ema = bt.indicators.EMA(
                self.data1.close, period=self.p.htf_ema_period
            )

        self._daily_closes: list[float] = []
        self._last_trading_date = None
        self._prev_day_close = 0.0
        self._regime_ema_val: float | None = None

        self._main_ref = None
        self._sl_ref = None
        self._tp_ref = None
        self._sl_order = None
        self._tp_order = None
        self._pending_dir = None
        self._signal_atr = 0.0
        self._order_pending = False
        self._current_qty = 0

        self._active_trade: dict | None = None
        self._trade_log: list[dict] = []

    # ── Helpers ───────────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        if self.p.printlog:
            dt = self.data.datetime.datetime(0)
            print(f"  {dt:%Y-%m-%d %H:%M}  {msg}")

    def _calc_qty(self, atr_val: float) -> int:
        if self.p.trade_qty > 0:
            return self.p.trade_qty
        capital = self.broker.getvalue()
        risk_qty = int(capital * self.p.risk_pct / atr_val)
        max_qty = int(capital * self.p.leverage_cap / self.data.close[0])
        return max(min(risk_qty, max_qty), 1)

    def _update_regime(self) -> None:
        """Maintain a running daily EMA from end-of-day closes."""
        cur = self.data.datetime.date(0)
        if cur != self._last_trading_date:
            if self._last_trading_date is not None and self._prev_day_close > 0:
                self._daily_closes.append(self._prev_day_close)
                p = self.p.regime_ema_period
                if len(self._daily_closes) >= p:
                    if self._regime_ema_val is None:
                        self._regime_ema_val = (
                            sum(self._daily_closes[-p:]) / p
                        )
                    else:
                        k = 2.0 / (p + 1)
                        self._regime_ema_val = (
                            self._daily_closes[-1] * k
                            + self._regime_ema_val * (1 - k)
                        )
            self._last_trading_date = cur
        self._prev_day_close = self.data.close[0]

    def _in_entry_window(self) -> bool:
        t = self.data.datetime.time(0)
        morning = self.MORNING_OPEN <= t <= self.MORNING_CLOSE
        afternoon = self.AFTERNOON_OPEN <= t <= self.AFTERNOON_CLOSE
        return morning or afternoon

    def _in_afternoon(self) -> bool:
        t = self.data.datetime.time(0)
        return self.AFTERNOON_OPEN <= t <= self.AFTERNOON_CLOSE

    def _is_squareoff(self) -> bool:
        return self.data.datetime.time(0) >= self.SQUAREOFF

    def _cancel_bracket(self) -> None:
        for o in (self._sl_order, self._tp_order):
            if o is not None and o.alive():
                self.cancel(o)
        self._sl_order = None
        self._tp_order = None
        self._sl_ref = None
        self._tp_ref = None

    # ── Order / Trade Notifications ───────────────────────────────────────

    def notify_order(self, order):
        if order.status in (order.Submitted, order.Accepted):
            return

        ref = order.ref

        if order.status == order.Completed:
            side = "BUY" if order.isbuy() else "SELL"
            qty = self._current_qty

            # ── Entry fill → submit bracket ──
            if self._main_ref is not None and ref == self._main_ref:
                entry = order.executed.price
                atr = self._signal_atr

                if self._pending_dir == "long":
                    sl_price = entry - atr
                    tp_price = entry + atr * self.p.rr_multiplier
                    sl_ord = self.sell(
                        exectype=bt.Order.Stop, price=sl_price, size=qty,
                    )
                    tp_ord = self.sell(
                        exectype=bt.Order.Limit, price=tp_price, size=qty,
                    )
                else:
                    sl_price = entry + atr
                    tp_price = entry - atr * self.p.rr_multiplier
                    sl_ord = self.buy(
                        exectype=bt.Order.Stop, price=sl_price, size=qty,
                    )
                    tp_ord = self.buy(
                        exectype=bt.Order.Limit, price=tp_price, size=qty,
                    )

                self._sl_order = sl_ord
                self._tp_order = tp_ord
                self._sl_ref = sl_ord.ref
                self._tp_ref = tp_ord.ref

                if self._active_trade is not None:
                    self._active_trade["entry_time"] = self.data.datetime.datetime(0)
                    self._active_trade["entry_price"] = entry
                    self._active_trade["sl_price"] = sl_price
                    self._active_trade["tp_price"] = tp_price
                    self._active_trade["qty"] = qty

                self._log(
                    f"{side} ENTRY @ ₹{entry:.2f} x{qty}  "
                    f"SL=₹{sl_price:.2f}  TP=₹{tp_price:.2f}  ATR={atr:.2f}"
                )
                self._main_ref = None
                self._pending_dir = None
                self._order_pending = False
                return

            # ── SL hit ──
            if self._sl_ref is not None and ref == self._sl_ref:
                self._log(f"SL HIT  → {side} @ ₹{order.executed.price:.2f}")
                if self._active_trade is not None:
                    self._active_trade["exit_time"] = self.data.datetime.datetime(0)
                    self._active_trade["exit_price"] = order.executed.price
                    self._active_trade["exit_reason"] = "SL"
                if self._tp_order is not None and self._tp_order.alive():
                    self.cancel(self._tp_order)
                self._sl_order = None
                self._tp_order = None
                self._sl_ref = None
                self._tp_ref = None
                return

            # ── TP hit ──
            if self._tp_ref is not None and ref == self._tp_ref:
                self._log(f"TP HIT  → {side} @ ₹{order.executed.price:.2f}")
                if self._active_trade is not None:
                    self._active_trade["exit_time"] = self.data.datetime.datetime(0)
                    self._active_trade["exit_price"] = order.executed.price
                    self._active_trade["exit_reason"] = "TP"
                if self._sl_order is not None and self._sl_order.alive():
                    self.cancel(self._sl_order)
                self._tp_order = None
                self._sl_order = None
                self._tp_ref = None
                self._sl_ref = None
                return

            # ── Squareoff / other close ──
            if self._active_trade is not None:
                self._active_trade["exit_time"] = self.data.datetime.datetime(0)
                self._active_trade["exit_price"] = order.executed.price
                self._active_trade.setdefault("exit_reason", "SQUAREOFF")
            self._log(f"CLOSE   → {side} @ ₹{order.executed.price:.2f}")

        elif order.status in (order.Canceled, order.Margin, order.Rejected):
            if self._main_ref is not None and ref == self._main_ref:
                self._log(f"ORDER REJECTED/CANCELED (status={order.status})")
                self._main_ref = None
                self._pending_dir = None
                self._order_pending = False
                self._active_trade = None

    def notify_trade(self, trade):
        if trade.isclosed:
            self._log(
                f"TRADE CLOSED  PnL=₹{trade.pnl:.2f}  Net=₹{trade.pnlcomm:.2f}"
            )
            if self._active_trade is not None:
                self._active_trade["pnl"] = trade.pnl
                self._active_trade["pnlcomm"] = trade.pnlcomm
                self._active_trade["bars"] = trade.barlen
                self._trade_log.append(dict(self._active_trade))
                self._active_trade = None

    # ── Core Logic ────────────────────────────────────────────────────────

    def next(self):
        self._update_regime()

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

        if self.p.use_htf_filter:
            allow_long = allow_long and close > self.htf_ema[0]
            allow_short = allow_short and close < self.htf_ema[0]

        vol_thresh = vol_avg * self.p.vol_mult

        # ── Long Entry ────────────────────────────────────────────────
        if (
            allow_long
            and close > vwap
            and cross > 0
            and vol > vol_thresh
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

        # ── Short Entry ───────────────────────────────────────────────
        elif (
            allow_short
            and close < vwap
            and cross < 0
            and vol > vol_thresh
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

    def stop(self):
        if not self._trade_log:
            self._log("NO TRADES EXECUTED")
            return
        wins = [t for t in self._trade_log if t["pnlcomm"] > 0]
        total = len(self._trade_log)
        self._log(
            f"DONE — {total} trades, {len(wins)} wins "
            f"({len(wins) / total * 100:.0f}%)"
        )

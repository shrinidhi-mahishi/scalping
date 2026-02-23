"""
Enhanced Strategy with Alpha-Generating Improvements

Extends IntradayScalpingStrategy with 7 independently toggleable improvements:

  1. VWAP Extension Limit  — reject signals when price > VWAP ± 1.5×ATR
  2. Volume Climax Cap     — reject exhaustion volume (> 3.5× SMA)
  3. Gap-Day Filter        — suppress signals on gap-up/down mornings
  4. PDH/PDL Filter        — respect previous day's high/low as S/R
  5. Index Alignment       — align with Nifty 50 VWAP direction
  6. Scale-Outs            — partial TP at 1R, SL→BE, final TP at 2.5R
  7. RSI Bands             — tighter momentum gate (already parameterized)

Usage:
  from strategy_enhanced import EnhancedStrategy

  # Single filter
  cerebro.addstrategy(EnhancedStrategy, vwap_ext_atr=1.5)

  # All filters
  cerebro.addstrategy(EnhancedStrategy,
      long_rsi_low=50, long_rsi_high=75,
      short_rsi_low=25, short_rsi_high=50,
      vwap_ext_atr=1.5, vol_cap_mult=3.5,
      gap_threshold=0.015, pdhl_buffer_atr=0.2,
  )

  # Scale-out
  cerebro.addstrategy(EnhancedStrategy, use_scale_out=True)
"""

import backtrader as bt

from strategy import IntradayScalpingStrategy, SessionVWAP


class EnhancedStrategy(IntradayScalpingStrategy):

    params = (
        ("vwap_ext_atr", 0),        # max |close-vwap|/ATR; 0 = disabled
        ("vol_cap_mult", 0),        # max vol / vol_sma; 0 = disabled
        ("gap_threshold", 0),       # gap % to block signals; 0 = disabled
        ("pdhl_buffer_atr", 0),     # PDH/PDL buffer in ATR units; 0 = disabled
        ("use_index_filter", False),
        ("use_scale_out", False),
        ("scale_r1", 1.0),         # TP1 at scale_r1 × ATR
        ("scale_pct", 0.5),        # fraction of qty to close at TP1
        ("scale_r2", 2.5),         # TP2 at scale_r2 × ATR
    )

    def __init__(self):
        super().__init__()

        self._prev_day_high = 0.0
        self._prev_day_low = float("inf")
        self._cur_day_high = 0.0
        self._cur_day_low = float("inf")
        self._pdhl_ready = False

        self._gap_long_blocked = False
        self._gap_short_blocked = False

        if self.p.use_index_filter and len(self.datas) >= 3:
            self.index_vwap = SessionVWAP(self.data2)

        self._scale_phase = None
        self._tp1_ref = None
        self._tp2_ref = None
        self._full_qty = 0
        self._partial_qty = 0

    # ── Extended regime tracker (adds PDH/PDL + gap) ───────────────────

    def _update_regime(self):
        cur = self.data.datetime.date(0)

        if cur != self._last_trading_date:
            if self._last_trading_date is not None:
                if self._prev_day_close > 0:
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

                if self._cur_day_high > 0:
                    self._prev_day_high = self._cur_day_high
                    self._prev_day_low = self._cur_day_low
                    self._pdhl_ready = True

                if self.p.gap_threshold > 0 and self._prev_day_close > 0:
                    gap = (
                        (self.data.open[0] - self._prev_day_close)
                        / self._prev_day_close
                    )
                    self._gap_long_blocked = gap > self.p.gap_threshold
                    self._gap_short_blocked = gap < -self.p.gap_threshold
                else:
                    self._gap_long_blocked = False
                    self._gap_short_blocked = False

            self._cur_day_high = 0.0
            self._cur_day_low = float("inf")
            self._last_trading_date = cur

        self._cur_day_high = max(self._cur_day_high, self.data.high[0])
        self._cur_day_low = min(self._cur_day_low, self.data.low[0])
        self._prev_day_close = self.data.close[0]

    # ── Bracket cancel with scale-out cleanup ──────────────────────────

    def _cancel_bracket(self):
        super()._cancel_bracket()
        self._tp1_ref = None
        self._tp2_ref = None
        self._scale_phase = None

    # ── Scale-Out order management ─────────────────────────────────────

    def notify_order(self, order):
        if not self.p.use_scale_out:
            return super().notify_order(order)

        if order.status in (order.Submitted, order.Accepted):
            return

        ref = order.ref

        if order.status == order.Completed:
            side = "BUY" if order.isbuy() else "SELL"

            # ── Entry fill → SL (full qty) + TP1 (partial qty) ──
            if self._main_ref is not None and ref == self._main_ref:
                entry = order.executed.price
                atr = self._signal_atr
                full = self._full_qty
                partial = self._partial_qty

                if self._pending_dir == "long":
                    sl_p = entry - atr
                    tp1_p = entry + atr * self.p.scale_r1
                    sl_ord = self.sell(
                        exectype=bt.Order.Stop, price=sl_p, size=full,
                    )
                    tp1_ord = self.sell(
                        exectype=bt.Order.Limit, price=tp1_p, size=partial,
                    )
                else:
                    sl_p = entry + atr
                    tp1_p = entry - atr * self.p.scale_r1
                    sl_ord = self.buy(
                        exectype=bt.Order.Stop, price=sl_p, size=full,
                    )
                    tp1_ord = self.buy(
                        exectype=bt.Order.Limit, price=tp1_p, size=partial,
                    )

                self._sl_order = sl_ord
                self._sl_ref = sl_ord.ref
                self._tp_order = tp1_ord
                self._tp_ref = tp1_ord.ref
                self._tp1_ref = tp1_ord.ref
                self._scale_phase = "full"

                if self._active_trade is not None:
                    self._active_trade.update({
                        "entry_time": self.data.datetime.datetime(0),
                        "entry_price": entry,
                        "sl_price": sl_p,
                        "tp_price": tp1_p,
                        "qty": full,
                    })

                self._log(
                    f"{side} ENTRY @ ₹{entry:.2f} x{full}  "
                    f"SL=₹{sl_p:.2f}  TP1=₹{tp1_p:.2f} (x{partial})"
                )
                self._main_ref = None
                self._pending_dir = None
                self._order_pending = False
                return

            # ── SL hit (full or breakeven) ──
            if self._sl_ref is not None and ref == self._sl_ref:
                reason = (
                    "SL" if self._scale_phase == "full" else "SL_BREAKEVEN"
                )
                self._log(
                    f"{reason} HIT → {side} @ ₹{order.executed.price:.2f}"
                )
                if self._active_trade is not None:
                    self._active_trade["exit_time"] = (
                        self.data.datetime.datetime(0)
                    )
                    self._active_trade["exit_price"] = order.executed.price
                    self._active_trade["exit_reason"] = reason

                if self._tp_order is not None and self._tp_order.alive():
                    self.cancel(self._tp_order)

                self._sl_order = self._tp_order = None
                self._sl_ref = self._tp_ref = None
                self._tp1_ref = self._tp2_ref = None
                self._scale_phase = None
                return

            # ── TP1 hit → cancel old SL, submit BE + TP2 ──
            if self._tp1_ref is not None and ref == self._tp1_ref:
                self._log(
                    f"TP1 HIT → {side} @ ₹{order.executed.price:.2f}"
                )

                if self._sl_order is not None and self._sl_order.alive():
                    self.cancel(self._sl_order)

                remaining = self._full_qty - self._partial_qty
                if remaining <= 0:
                    if self._active_trade is not None:
                        self._active_trade["exit_time"] = (
                            self.data.datetime.datetime(0)
                        )
                        self._active_trade["exit_price"] = (
                            order.executed.price
                        )
                        self._active_trade["exit_reason"] = "TP"
                    self._sl_order = self._tp_order = None
                    self._sl_ref = self._tp_ref = None
                    self._tp1_ref = self._tp2_ref = None
                    self._scale_phase = None
                    return

                entry = self._active_trade["entry_price"]
                atr = self._signal_atr

                if self._active_trade["direction"] == "LONG":
                    be_p = entry
                    tp2_p = entry + atr * self.p.scale_r2
                    sl_ord = self.sell(
                        exectype=bt.Order.Stop, price=be_p, size=remaining,
                    )
                    tp2_ord = self.sell(
                        exectype=bt.Order.Limit, price=tp2_p, size=remaining,
                    )
                else:
                    be_p = entry
                    tp2_p = entry - atr * self.p.scale_r2
                    sl_ord = self.buy(
                        exectype=bt.Order.Stop, price=be_p, size=remaining,
                    )
                    tp2_ord = self.buy(
                        exectype=bt.Order.Limit, price=tp2_p, size=remaining,
                    )

                self._sl_order = sl_ord
                self._sl_ref = sl_ord.ref
                self._tp_order = tp2_ord
                self._tp_ref = tp2_ord.ref
                self._tp2_ref = tp2_ord.ref
                self._tp1_ref = None
                self._scale_phase = "partial"

                if self._active_trade is not None:
                    self._active_trade["tp1_hit"] = True
                    self._active_trade["tp1_price"] = order.executed.price
                    self._active_trade["sl_price"] = be_p
                    self._active_trade["tp_price"] = tp2_p

                self._log(
                    f"SCALE OUT  SL→BE ₹{be_p:.2f}  "
                    f"TP2=₹{tp2_p:.2f} (x{remaining})"
                )
                return

            # ── TP2 hit → full close ──
            if self._tp2_ref is not None and ref == self._tp2_ref:
                self._log(
                    f"TP2 HIT → {side} @ ₹{order.executed.price:.2f}"
                )
                if self._active_trade is not None:
                    self._active_trade["exit_time"] = (
                        self.data.datetime.datetime(0)
                    )
                    self._active_trade["exit_price"] = order.executed.price
                    self._active_trade["exit_reason"] = "TP2"

                if self._sl_order is not None and self._sl_order.alive():
                    self.cancel(self._sl_order)

                self._sl_order = self._tp_order = None
                self._sl_ref = self._tp_ref = None
                self._tp1_ref = self._tp2_ref = None
                self._scale_phase = None
                return

            # ── Squareoff / other close ──
            if self._active_trade is not None:
                self._active_trade["exit_time"] = (
                    self.data.datetime.datetime(0)
                )
                self._active_trade["exit_price"] = order.executed.price
                self._active_trade.setdefault("exit_reason", "SQUAREOFF")
            self._log(f"CLOSE → {side} @ ₹{order.executed.price:.2f}")

        elif order.status in (order.Canceled, order.Margin, order.Rejected):
            if self._main_ref is not None and ref == self._main_ref:
                self._log(
                    f"ORDER REJECTED/CANCELED (status={order.status})"
                )
                self._main_ref = None
                self._pending_dir = None
                self._order_pending = False
                self._active_trade = None
                self._scale_phase = None

    # ── Core logic with enhanced filters ───────────────────────────────

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

        # ── Base filters (regime + HTF) ──
        if self.p.use_regime_filter and self._regime_ema_val is not None:
            allow_long = close > self._regime_ema_val
            allow_short = close < self._regime_ema_val
        else:
            allow_long = True
            allow_short = True

        if self.p.use_htf_filter and hasattr(self, "htf_ema"):
            allow_long = allow_long and close > self.htf_ema[0]
            allow_short = allow_short and close < self.htf_ema[0]

        # ── VWAP Extension Limit ──
        if self.p.vwap_ext_atr > 0 and atr_val > 0:
            if abs(close - vwap) / atr_val > self.p.vwap_ext_atr:
                allow_long = False
                allow_short = False

        # ── Gap-Day Filter (morning session only) ──
        if self.p.gap_threshold > 0:
            t = self.data.datetime.time(0)
            if self.MORNING_OPEN <= t <= self.MORNING_CLOSE:
                if self._gap_long_blocked:
                    allow_long = False
                if self._gap_short_blocked:
                    allow_short = False

        # ── PDH/PDL Filter ──
        if self.p.pdhl_buffer_atr > 0 and self._pdhl_ready and atr_val > 0:
            buf = self.p.pdhl_buffer_atr * atr_val
            # Don't LONG into PDH resistance (allow if broken through)
            if close < self._prev_day_high and close > self._prev_day_high - buf:
                allow_long = False
            # Don't SHORT into PDL support (allow if broken through)
            if close > self._prev_day_low and close < self._prev_day_low + buf:
                allow_short = False

        # ── Index VWAP Filter ──
        if self.p.use_index_filter and hasattr(self, "index_vwap"):
            idx_c = self.data2.close[0]
            idx_v = self.index_vwap[0]
            if idx_v > 0:
                allow_long = allow_long and idx_c > idx_v
                allow_short = allow_short and idx_c < idx_v

        # ── Volume check with optional climax cap ──
        vol_thresh = vol_avg * self.p.vol_mult
        vol_ok = vol > vol_thresh
        if self.p.vol_cap_mult > 0 and vol_avg > 0:
            vol_ok = vol_ok and vol < vol_avg * self.p.vol_cap_mult

        # ── Long Entry ──
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
            if self.p.use_scale_out:
                self._scale_phase = "full"
                self._full_qty = qty
                self._partial_qty = max(int(qty * self.p.scale_pct), 1)
            self._log(
                f"LONG SIGNAL   Close={close:.2f}  VWAP={vwap:.2f}  "
                f"RSI={rsi_val:.1f}  ATR={atr_val:.2f}  Qty={qty}"
            )

        # ── Short Entry ──
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
            if self.p.use_scale_out:
                self._scale_phase = "full"
                self._full_qty = qty
                self._partial_qty = max(int(qty * self.p.scale_pct), 1)
            self._log(
                f"SHORT SIGNAL  Close={close:.2f}  VWAP={vwap:.2f}  "
                f"RSI={rsi_val:.1f}  ATR={atr_val:.2f}  Qty={qty}"
            )

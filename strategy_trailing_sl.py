"""
Trailing Stop-Loss Strategy Variants for Backtesting

Extends IntradayScalpingStrategy to dynamically adjust the SL order
as price moves in the trade's favor, locking in profits.

Trail modes:
  atr_trail      — SL trails 1×ATR below peak price (LONG) or above trough (SHORT)
  candle_trail   — SL moves to prev candle's low (LONG) or high (SHORT)
  pct_trail      — SL trails at a fixed % below peak (default 0.5%)
  breakeven_atr  — move SL to breakeven after 1×ATR profit, then trail with ATR

Optional early exit (exit_mode param) can be combined with any trail mode.

Usage:
  from strategy_trailing_sl import TrailingSLStrategy
  cerebro.addstrategy(TrailingSLStrategy, trail_mode="atr_trail", printlog=False)
"""

import backtrader as bt

from strategy import IntradayScalpingStrategy


class TrailingSLStrategy(IntradayScalpingStrategy):

    params = (
        ("trail_mode", "atr_trail"),
        ("trail_pct", 0.005),
        ("max_trade_bars", 15),
        ("min_profit_atr", 0.2),
        ("exit_mode", ""),
    )

    def __init__(self):
        super().__init__()
        self._reset_trail_state()

    def _reset_trail_state(self):
        self._entry_price: float = 0.0
        self._current_sl: float = 0.0
        self._trail_high: float = 0.0
        self._trail_low: float = float("inf")
        self._breakeven_hit: bool = False
        self._sl_trailed: bool = False
        self._entry_bar: int = 0

    # ── Order notification — capture entry state ──────────────────────────

    def notify_order(self, order):
        was_entry_fill = (
            self._main_ref is not None
            and order.ref == self._main_ref
            and order.status == order.Completed
        )
        was_trailed_sl = (
            self._sl_ref is not None
            and order.ref == self._sl_ref
            and order.status == order.Completed
            and self._sl_trailed
        )

        super().notify_order(order)

        if was_entry_fill:
            self._entry_price = order.executed.price
            sl = self._active_trade["sl_price"] if self._active_trade else 0
            self._current_sl = sl
            self._trail_high = self.data.high[0]
            self._trail_low = self.data.low[0]
            self._breakeven_hit = False
            self._sl_trailed = False
            self._entry_bar = len(self)

        if was_trailed_sl and self._active_trade is not None:
            self._active_trade["exit_reason"] = "TRAIL_SL"

    def notify_trade(self, trade):
        super().notify_trade(trade)
        if trade.isclosed:
            self._reset_trail_state()

    # ── SL movement helper ────────────────────────────────────────────────

    def _move_sl(self, new_price: float) -> None:
        if self._sl_order is not None and self._sl_order.alive():
            self.cancel(self._sl_order)

        qty = self._current_qty
        if self.position.size > 0:
            new_ord = self.sell(
                exectype=bt.Order.Stop, price=new_price, size=qty
            )
        else:
            new_ord = self.buy(
                exectype=bt.Order.Stop, price=new_price, size=qty
            )

        self._sl_order = new_ord
        self._sl_ref = new_ord.ref
        self._sl_trailed = True
        self._current_sl = new_price
        if self._active_trade is not None:
            self._active_trade["sl_price"] = new_price

    # ── Trailing logic per mode ───────────────────────────────────────────

    def _update_trailing_sl(self) -> None:
        is_long = self.position.size > 0
        atr = self._signal_atr

        if is_long:
            self._trail_high = max(self._trail_high, self.data.high[0])
        else:
            self._trail_low = min(self._trail_low, self.data.low[0])

        mode = self.p.trail_mode

        if mode == "atr_trail":
            self._trail_atr(is_long, atr)
        elif mode == "candle_trail":
            self._trail_candle(is_long)
        elif mode == "pct_trail":
            self._trail_pct(is_long)
        elif mode == "breakeven_atr":
            self._trail_breakeven_atr(is_long, atr)

    def _trail_atr(self, is_long: bool, atr: float) -> None:
        if is_long:
            new_sl = self._trail_high - atr
            if new_sl > self._current_sl:
                self._move_sl(new_sl)
        else:
            new_sl = self._trail_low + atr
            if new_sl < self._current_sl:
                self._move_sl(new_sl)

    def _trail_candle(self, is_long: bool) -> None:
        if is_long:
            new_sl = self.data.low[-1]
            if new_sl > self._current_sl:
                self._move_sl(new_sl)
        else:
            new_sl = self.data.high[-1]
            if new_sl < self._current_sl:
                self._move_sl(new_sl)

    def _trail_pct(self, is_long: bool) -> None:
        pct = self.p.trail_pct
        if is_long:
            new_sl = self._trail_high * (1.0 - pct)
            if new_sl > self._current_sl:
                self._move_sl(new_sl)
        else:
            new_sl = self._trail_low * (1.0 + pct)
            if new_sl < self._current_sl:
                self._move_sl(new_sl)

    def _trail_breakeven_atr(self, is_long: bool, atr: float) -> None:
        if is_long:
            gain = self._trail_high - self._entry_price
            if not self._breakeven_hit and gain >= atr:
                self._move_sl(self._entry_price)
                self._breakeven_hit = True
            elif self._breakeven_hit:
                new_sl = self._trail_high - atr
                if new_sl > self._current_sl:
                    self._move_sl(new_sl)
        else:
            gain = self._entry_price - self._trail_low
            if not self._breakeven_hit and gain >= atr:
                self._move_sl(self._entry_price)
                self._breakeven_hit = True
            elif self._breakeven_hit:
                new_sl = self._trail_low + atr
                if new_sl < self._current_sl:
                    self._move_sl(new_sl)

    # ── Optional early exit (for combined mode) ───────────────────────────

    def _check_early_exit(self) -> bool:
        mode = self.p.exit_mode
        if not mode:
            return False

        is_long = self.position.size > 0

        if mode == "ema_reversal":
            if is_long and self.ema_cross[0] < 0:
                return True
            if not is_long and self.ema_cross[0] > 0:
                return True

        elif mode == "vwap_reversal":
            c = self.data.close[0]
            if is_long and c < self.vwap[0]:
                return True
            if not is_long and c > self.vwap[0]:
                return True

        elif mode == "rsi_extreme":
            rsi_val = self.rsi[0]
            if is_long and rsi_val < self.p.long_rsi_low:
                return True
            if not is_long and rsi_val > self.p.short_rsi_high:
                return True

        elif mode == "time_based":
            if self._entry_bar > 0:
                bars = len(self) - self._entry_bar
                if bars >= self.p.max_trade_bars:
                    return True

        elif mode == "stagnation":
            if self._entry_bar > 0 and self._active_trade:
                bars = len(self) - self._entry_bar
                if bars >= self.p.max_trade_bars:
                    entry_price = self._active_trade.get("entry_price", 0)
                    if entry_price > 0:
                        close = self.data.close[0]
                        unrealized = (close - entry_price) if is_long else (entry_price - close)
                        if unrealized < self.p.min_profit_atr * self._signal_atr:
                            return True

        return False

    # ── Overridden next() ─────────────────────────────────────────────────

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

            self._update_trailing_sl()

        super().next()

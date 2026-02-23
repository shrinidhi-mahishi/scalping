"""
Early Exit Strategy Variants for Backtesting

Extends IntradayScalpingStrategy to exit trades before SL when conditions
indicate the stock is not behaving as expected.

Exit modes:
  ema_reversal  — exit if EMA(9) crosses back against trade direction
  vwap_reversal — exit if price crosses back through VWAP against direction
  rsi_extreme   — exit if RSI moves outside the entry range
  time_based    — exit if TP not hit within N candles (default 15 = 45 min)
  stagnation    — exit if profit < 0.2×ATR after 8 bars (momentum fakeout)

Usage:
  from strategy_early_exit import EarlyExitStrategy
  cerebro.addstrategy(EarlyExitStrategy, exit_mode="ema_reversal", printlog=False)
"""

from strategy import IntradayScalpingStrategy


class EarlyExitStrategy(IntradayScalpingStrategy):

    params = (
        ("exit_mode", "ema_reversal"),
        ("max_trade_bars", 15),
        ("min_profit_atr", 0.2),
    )

    def __init__(self):
        super().__init__()
        self._entry_bar: int = 0

    def notify_order(self, order):
        was_entry_fill = (
            self._main_ref is not None
            and order.ref == self._main_ref
            and order.status == order.Completed
        )
        super().notify_order(order)
        if was_entry_fill:
            self._entry_bar = len(self)

    # ── Early exit logic ──────────────────────────────────────────────────

    def _check_early_exit(self) -> bool:
        mode = self.p.exit_mode
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

        super().next()

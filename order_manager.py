"""
Angel One SmartAPI order placement for automated trading.

Places NORMAL INTRADAY limit orders for entry, then separate
STOPLOSS_LIMIT and LIMIT orders for SL and TP protection.

Entry buffer: small ATR-capped rupee amount (not percentage).
Auto-cancel: unfilled entry orders are cancelled after 60s or next bar close.
Exchange verification: SL/TP orders are checked for exchange rejection after placement.
"""

import logging
import time as _time

from fetch_data import AngelOneClient

flog = logging.getLogger("live_signals")

TICK_BANDS = [
    (20_001, 5.00),
    (10_001, 1.00),
    (5_001,  0.50),
    (501,    0.10),
    (0,      0.05),
]
FILL_TIMEOUT_SEC = 60
FILL_POLL_SEC = 3
FINAL_STATUS_WAIT_SEC = 8
FINAL_STATUS_POLL_SEC = 1

TRADING_SYMBOL_SUFFIX = {
    "M&M": "M%26M-EQ",
}


def _trading_symbol(sym: str) -> str:
    """Convert internal symbol name to Angel One trading symbol."""
    if sym in TRADING_SYMBOL_SUFFIX:
        return TRADING_SYMBOL_SUFFIX[sym]
    return f"{sym}-EQ"


def _tick_for_price(price: float) -> float:
    """Return the correct NSE tick size based on price-linked band (revised Apr 2025)."""
    p = abs(price)
    for threshold, tick in TICK_BANDS:
        if p >= threshold:
            return tick
    return 0.05


def round_to_tick(price: float, tick: float | None = None) -> float:
    """Round a price to the nearest NSE tick size using Decimal for exactness."""
    from decimal import Decimal, ROUND_HALF_UP
    if tick is None:
        tick = _tick_for_price(price)
    d_tick = Decimal(str(tick))
    d_price = Decimal(str(price))
    rounded = (d_price / d_tick).quantize(Decimal('1'), rounding=ROUND_HALF_UP) * d_tick
    return float(rounded.quantize(Decimal('0.01')))


def _entry_buffer(atr: float) -> float:
    """Compute limit price buffer in rupees, capped by ATR."""
    return round_to_tick(max(0.10, min(0.50, 0.03 * atr)))


def place_bracket_order(
    conn,
    sym: str,
    direction: str,
    entry_price: float,
    sl_price: float,
    tp_price: float,
    qty: int,
    atr: float = 0.0,
    bar_close: float = 0.0,
) -> dict:
    """Place a NORMAL INTRADAY limit order for entry.

    Uses bar_close (if provided) as the reference for the limit price,
    since the trigger price may already be behind the market by the time
    the signal fires.

    Returns dict with keys: success, order_id, message, params
    """
    token = AngelOneClient.SYMBOL_TOKENS.get(sym)
    if not token:
        return {"success": False, "order_id": None,
                "message": f"Unknown symbol: {sym}", "params": None}

    trading_sym = _trading_symbol(sym)
    txn_type = "BUY" if direction == "LONG" else "SELL"

    buf = _entry_buffer(atr) if atr > 0 else 0.10
    ref_price = bar_close if bar_close > 0 else entry_price
    if direction == "LONG":
        limit_price = round_to_tick(ref_price + buf)
    else:
        limit_price = round_to_tick(ref_price - buf)

    params = {
        "variety": "NORMAL",
        "tradingsymbol": trading_sym,
        "symboltoken": token,
        "transactiontype": txn_type,
        "exchange": "NSE",
        "ordertype": "LIMIT",
        "producttype": "INTRADAY",
        "duration": "DAY",
        "price": f"{limit_price:.2f}",
        "quantity": str(qty),
        "scripconsent": "yes",
    }

    flog.info(
        "ORDER  %-11s  %s  qty=%d  limit=%.2f  buf=%.2f  SL=%.2f  TP=%.2f  sym=%s",
        sym, txn_type, qty, limit_price, buf, sl_price, tp_price, trading_sym,
    )

    try:
        raw_resp = conn._postRequest("api.order.place", params)
        flog.info("ORDER RAW RESP  %-11s  %s", sym, raw_resp)

        if raw_resp is None:
            flog.error("ORDER FAILED  %-11s  API returned None", sym)
            return {"success": False, "order_id": None,
                    "message": "API returned None — check static IP registration and BO eligibility",
                    "params": params}

        if not raw_resp.get("status"):
            msg = raw_resp.get("message", "Unknown error")
            error_code = raw_resp.get("errorcode", "")
            flog.error("ORDER FAILED  %-11s  %s (code=%s)", sym, msg, error_code)
            return {"success": False, "order_id": None,
                    "message": f"{msg} (code={error_code})", "params": params}

        data = raw_resp.get("data", {})
        order_id = data.get("orderid") if isinstance(data, dict) else None
        if order_id:
            flog.info("ORDER OK  %-11s  order_id=%s", sym, order_id)
            return {"success": True, "order_id": order_id,
                    "message": "Order placed", "params": params}

        flog.error("ORDER FAILED  %-11s  no orderid in response: %s", sym, raw_resp)
        return {"success": False, "order_id": None,
                "message": f"No orderid: {raw_resp}", "params": params}

    except Exception as exc:
        flog.error("ORDER ERROR  %-11s  %s", sym, exc)
        return {"success": False, "order_id": None,
                "message": str(exc), "params": params}


EMERGENCY_EXIT_SLIPPAGE_PCT = 0.005


def place_market_exit_order(
    conn,
    sym: str,
    direction: str,
    qty: int,
    ref_price: float = 0.0,
) -> dict:
    """Place an aggressive LIMIT order to flatten an existing position.

    Uses a wide buffer (0.5% of ref_price) to ensure immediate fill while
    complying with the SEBI/NSE prohibition on MARKET orders for algo trading
    (effective April 1, 2026).
    """
    token = AngelOneClient.SYMBOL_TOKENS.get(sym)
    if not token:
        return {"success": False, "order_id": None, "message": f"Unknown symbol: {sym}"}

    trading_sym = _trading_symbol(sym)
    exit_txn = "SELL" if direction == "LONG" else "BUY"

    if ref_price > 0:
        slippage = max(ref_price * EMERGENCY_EXIT_SLIPPAGE_PCT, _tick_for_price(ref_price) * 5)
        if direction == "LONG":
            limit_price = round_to_tick(ref_price - slippage)
        else:
            limit_price = round_to_tick(ref_price + slippage)
        order_type = "LIMIT"
    else:
        limit_price = 0.0
        order_type = "LIMIT"
        flog.warning("EXIT ORDER  %-11s  no ref_price provided — using best-effort limit", sym)

    params = {
        "variety": "NORMAL",
        "tradingsymbol": trading_sym,
        "symboltoken": token,
        "transactiontype": exit_txn,
        "exchange": "NSE",
        "ordertype": order_type,
        "producttype": "INTRADAY",
        "duration": "DAY",
        "price": f"{limit_price:.2f}",
        "quantity": str(qty),
        "scripconsent": "yes",
    }

    flog.info("EXIT ORDER  %-11s  %s  qty=%d  type=%s  limit=%.2f  ref=%.2f  slip=%.1f%%",
              sym, exit_txn, qty, order_type, limit_price, ref_price, EMERGENCY_EXIT_SLIPPAGE_PCT * 100)

    try:
        raw_resp = conn._postRequest("api.order.place", params)
        flog.info("EXIT ORDER RAW  %-11s  %s", sym, raw_resp)

        if raw_resp and raw_resp.get("status"):
            order_id = raw_resp.get("data", {}).get("orderid")
            if order_id:
                _time.sleep(1)
                exch = get_order_status(conn, order_id)
                exch_status = exch.get("orderstatus", "").lower() if exch else "unknown"
                if exch_status == "rejected":
                    reason = exch.get("text", "") if exch else ""
                    flog.error("EXIT ORDER REJECTED BY EXCHANGE  %-11s  %s", sym, reason)
                    return {"success": False, "order_id": order_id,
                            "message": f"Exchange rejected: {reason}"}
                flog.info("EXIT ORDER OK  %-11s  order_id=%s  exchange=%s", sym, order_id, exch_status)
                return {"success": True, "order_id": order_id, "message": "Exit order placed"}

        msg = raw_resp.get("message", "Unknown") if raw_resp else "None"
        flog.error("EXIT ORDER FAILED  %-11s  %s", sym, msg)
        return {"success": False, "order_id": None, "message": msg}

    except Exception as exc:
        flog.error("EXIT ORDER ERROR  %-11s  %s", sym, exc)
        return {"success": False, "order_id": None, "message": str(exc)}


def get_order_status(conn, order_id: str) -> dict | None:
    """Check the status of a placed order via the order book."""
    try:
        resp = conn.orderBook()
        if resp and resp.get("status") and resp.get("data"):
            for order in resp["data"]:
                if order.get("orderid") == order_id:
                    return order
    except Exception as exc:
        flog.warning("ORDER STATUS ERROR  %s: %s", order_id, exc)
    return None


def cancel_order(conn, order_id: str, variety: str = "NORMAL") -> bool:
    """Cancel an open order. Returns True if cancellation succeeded."""
    try:
        resp = conn.cancelOrder(order_id, variety)
        if resp:
            flog.info("ORDER CANCELLED  order_id=%s", order_id)
            return True
        flog.warning("ORDER CANCEL FAILED  order_id=%s  resp=%s", order_id, resp)
        return False
    except Exception as exc:
        flog.warning("ORDER CANCEL ERROR  order_id=%s  %s", order_id, exc)
        return False


def _confirm_terminal_order_state(
    conn,
    sym: str,
    order_id: str,
    *,
    filled_qty_hint: int = 0,
    fill_price_hint: float = 0.0,
    status_hint: str = "unknown",
    wait_sec: int = FINAL_STATUS_WAIT_SEC,
    poll_sec: int = FINAL_STATUS_POLL_SEC,
) -> dict:
    """Poll for a broker-confirmed final order state after timeout/cancel.

    This avoids telling the user "not filled" while the broker still shows the
    order as open or while a late fill/cancel acknowledgement is propagating.
    """
    deadline = _time.monotonic() + wait_sec
    last_status = status_hint or "unknown"
    last_filled_qty = int(filled_qty_hint or 0)
    last_fill_price = float(fill_price_hint or 0.0)

    while True:
        status = get_order_status(conn, order_id)
        if status is not None:
            order_status = str(status.get("orderstatus", "")).lower() or last_status
            filled_qty = int(status.get("filledshares", 0) or last_filled_qty)
            fill_price = float(status.get("averageprice", 0) or last_fill_price)

            last_status = order_status
            last_filled_qty = filled_qty
            last_fill_price = fill_price

            if order_status in ("complete", "filled"):
                flog.info(
                    "ORDER FINAL  %-11s  order_id=%s  status=%s  qty=%d  avg=%.2f",
                    sym, order_id, order_status, filled_qty, fill_price,
                )
                return {
                    "filled": True,
                    "filled_qty": filled_qty,
                    "fill_price": fill_price,
                    "status": order_status,
                    "cancelled": False,
                    "resolved": True,
                }

            if order_status == "cancelled":
                flog.info(
                    "ORDER FINAL  %-11s  order_id=%s  status=cancelled  qty=%d  avg=%.2f",
                    sym, order_id, filled_qty, fill_price,
                )
                return {
                    "filled": filled_qty > 0,
                    "filled_qty": filled_qty,
                    "fill_price": fill_price,
                    "status": "partial" if filled_qty > 0 else "cancelled",
                    "cancelled": True,
                    "resolved": True,
                }

            if order_status == "rejected":
                flog.info("ORDER FINAL  %-11s  order_id=%s  status=rejected", sym, order_id)
                return {
                    "filled": False,
                    "filled_qty": 0,
                    "fill_price": 0.0,
                    "status": "rejected",
                    "cancelled": False,
                    "resolved": True,
                }

        if _time.monotonic() >= deadline:
            break
        _time.sleep(poll_sec)

    flog.warning(
        "ORDER FINAL UNCERTAIN  %-11s  order_id=%s  status=%s  qty=%d  avg=%.2f",
        sym, order_id, last_status, last_filled_qty, last_fill_price,
    )
    return {
        "filled": last_filled_qty > 0,
        "filled_qty": last_filled_qty,
        "fill_price": last_fill_price,
        "status": last_status,
        "cancelled": False,
        "resolved": False,
    }


def place_sl_order(
    conn,
    sym: str,
    direction: str,
    sl_price: float,
    qty: int,
) -> dict:
    """Place a stop-loss order to protect a filled entry position."""
    token = AngelOneClient.SYMBOL_TOKENS.get(sym)
    if not token:
        return {"success": False, "order_id": None, "message": f"Unknown symbol: {sym}"}

    trading_sym = _trading_symbol(sym)
    sl_txn = "SELL" if direction == "LONG" else "BUY"
    trigger_price = round_to_tick(sl_price)
    sl_limit_buf = round_to_tick(max(0.10, abs(sl_price) * 0.002))
    if direction == "LONG":
        sl_limit = round_to_tick(trigger_price - sl_limit_buf)
    else:
        sl_limit = round_to_tick(trigger_price + sl_limit_buf)

    params = {
        "variety": "STOPLOSS",
        "tradingsymbol": trading_sym,
        "symboltoken": token,
        "transactiontype": sl_txn,
        "exchange": "NSE",
        "ordertype": "STOPLOSS_LIMIT",
        "producttype": "INTRADAY",
        "duration": "DAY",
        "price": f"{sl_limit:.2f}",
        "triggerprice": f"{trigger_price:.2f}",
        "quantity": str(qty),
        "scripconsent": "yes",
    }

    flog.info("SL ORDER  %-11s  %s  qty=%d  trigger=%.2f  limit=%.2f", sym, sl_txn, qty, trigger_price, sl_limit)

    try:
        raw_resp = conn._postRequest("api.order.place", params)
        flog.info("SL ORDER RAW  %-11s  %s", sym, raw_resp)

        if raw_resp and raw_resp.get("status"):
            order_id = raw_resp.get("data", {}).get("orderid")
            if order_id:
                _time.sleep(1)
                exch = get_order_status(conn, order_id)
                exch_status = exch.get("orderstatus", "").lower() if exch else "unknown"
                if exch_status == "rejected":
                    reason = exch.get("text", "") if exch else ""
                    flog.error("SL ORDER REJECTED BY EXCHANGE  %-11s  %s", sym, reason)
                    return {"success": False, "order_id": order_id,
                            "message": f"Exchange rejected: {reason}"}
                flog.info("SL ORDER OK  %-11s  order_id=%s  exchange=%s", sym, order_id, exch_status)
                return {"success": True, "order_id": order_id, "message": "SL placed"}

        msg = raw_resp.get("message", "Unknown") if raw_resp else "None"
        flog.error("SL ORDER FAILED  %-11s  %s", sym, msg)
        return {"success": False, "order_id": None, "message": msg}

    except Exception as exc:
        flog.error("SL ORDER ERROR  %-11s  %s", sym, exc)
        return {"success": False, "order_id": None, "message": str(exc)}


def place_tp_order(
    conn,
    sym: str,
    direction: str,
    tp_price: float,
    qty: int,
) -> dict:
    """Place a take-profit limit order to close a filled position."""
    token = AngelOneClient.SYMBOL_TOKENS.get(sym)
    if not token:
        return {"success": False, "order_id": None, "message": f"Unknown symbol: {sym}"}

    trading_sym = _trading_symbol(sym)
    tp_txn = "SELL" if direction == "LONG" else "BUY"
    limit_price = round_to_tick(tp_price)

    params = {
        "variety": "NORMAL",
        "tradingsymbol": trading_sym,
        "symboltoken": token,
        "transactiontype": tp_txn,
        "exchange": "NSE",
        "ordertype": "LIMIT",
        "producttype": "INTRADAY",
        "duration": "DAY",
        "price": f"{limit_price:.2f}",
        "quantity": str(qty),
        "scripconsent": "yes",
    }

    flog.info("TP ORDER  %-11s  %s  qty=%d  limit=%.2f", sym, tp_txn, qty, limit_price)

    try:
        raw_resp = conn._postRequest("api.order.place", params)
        flog.info("TP ORDER RAW  %-11s  %s", sym, raw_resp)

        if raw_resp and raw_resp.get("status"):
            order_id = raw_resp.get("data", {}).get("orderid")
            if order_id:
                _time.sleep(1)
                exch = get_order_status(conn, order_id)
                exch_status = exch.get("orderstatus", "").lower() if exch else "unknown"
                if exch_status == "rejected":
                    reason = exch.get("text", "") if exch else ""
                    flog.error("TP ORDER REJECTED BY EXCHANGE  %-11s  %s", sym, reason)
                    return {"success": False, "order_id": order_id,
                            "message": f"Exchange rejected: {reason}"}
                flog.info("TP ORDER OK  %-11s  order_id=%s  exchange=%s", sym, order_id, exch_status)
                return {"success": True, "order_id": order_id, "message": "TP placed"}

        msg = raw_resp.get("message", "Unknown") if raw_resp else "None"
        flog.error("TP ORDER FAILED  %-11s  %s", sym, msg)
        return {"success": False, "order_id": None, "message": msg}

    except Exception as exc:
        flog.error("TP ORDER ERROR  %-11s  %s", sym, exc)
        return {"success": False, "order_id": None, "message": str(exc)}


def wait_for_fill_or_cancel(
    conn,
    sym: str,
    order_id: str,
    bar_expiry_ts=None,
    timeout_sec: int = FILL_TIMEOUT_SEC,
) -> dict:
    """Poll order status and cancel if not filled within timeout or bar expiry.

    Returns dict with: filled, filled_qty, fill_price, status, cancelled, resolved
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo
    IST = ZoneInfo("Asia/Kolkata")

    start = _time.monotonic()
    filled_qty = 0
    fill_price = 0.0
    final_status = "unknown"

    while True:
        elapsed = _time.monotonic() - start
        now_ist = datetime.now(IST).replace(tzinfo=None)

        timed_out = elapsed >= timeout_sec
        bar_expired = bar_expiry_ts is not None and now_ist >= bar_expiry_ts

        if timed_out or bar_expired:
            reason = "timeout" if timed_out else "bar_expired"
            flog.info("ORDER EXPIRY  %-11s  %s  order_id=%s  elapsed=%.0fs  filled=%d",
                       sym, reason, order_id, elapsed, filled_qty)
            break

        status = get_order_status(conn, order_id)
        if status is None:
            _time.sleep(FILL_POLL_SEC)
            continue

        order_status = str(status.get("orderstatus", "")).lower()
        filled_qty = int(status.get("filledshares", 0) or 0)
        fill_price = float(status.get("averageprice", 0) or 0)
        final_status = order_status

        if order_status in ("complete", "filled"):
            flog.info("ORDER FILLED  %-11s  order_id=%s  qty=%d  avg=%.2f  in %.1fs",
                       sym, order_id, filled_qty, fill_price, elapsed)
            return {"filled": True, "filled_qty": filled_qty,
                    "fill_price": fill_price, "status": order_status, "cancelled": False, "resolved": True}

        if order_status in ("rejected", "cancelled"):
            flog.warning("ORDER %s  %-11s  order_id=%s", order_status.upper(), sym, order_id)
            return {"filled": False, "filled_qty": 0,
                    "fill_price": 0, "status": order_status, "cancelled": order_status == "cancelled", "resolved": True}

        _time.sleep(FILL_POLL_SEC)

    if filled_qty > 0:
        flog.info("ORDER PARTIAL  %-11s  filled=%d  cancelling remainder", sym, filled_qty)

    cancel_requested = cancel_order(conn, order_id)
    confirmed = _confirm_terminal_order_state(
        conn,
        sym,
        order_id,
        filled_qty_hint=filled_qty,
        fill_price_hint=fill_price,
        status_hint=final_status,
    )
    confirmed["cancelled"] = bool(confirmed.get("cancelled")) or cancel_requested
    return confirmed

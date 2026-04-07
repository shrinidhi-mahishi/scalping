"""
Broker-aligned P&L report for Angel One intraday trading.

Modes:
1. Current day (default): uses Angel One APIs directly
2. Date range: uses Angel One for *today* if included, and reconstructs
   prior realized trades from archived live logs in a broker-aligned way

Usage:
    .venv/bin/python broker_pnl_report.py
    .venv/bin/python broker_pnl_report.py --date 2026-04-02
    .venv/bin/python broker_pnl_report.py --start-date 2026-03-23 --end-date 2026-04-02
    .venv/bin/python broker_pnl_report.py --date 2026-04-02 --csv broker_report_2026-04-02.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict, deque
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from fetch_data import AngelOneClient

BASE_DIR = Path(__file__).resolve().parent
LIVE_DIR = BASE_DIR / "logs" / "live_signals"
SIGNAL_DIR = BASE_DIR / "logs" / "signals"


def _money(x: float) -> str:
    return f"₹{x:,.2f}"


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _load_signal_rows(day: date) -> dict[str, deque[dict]]:
    path = SIGNAL_DIR / f"signals_{day:%Y-%m-%d}.csv"
    by_sym: dict[str, deque[dict]] = defaultdict(deque)
    if not path.exists():
        return by_sym
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["price"] = float(row.get("price", 0) or 0)
            row["sl"] = float(row.get("sl", 0) or 0)
            row["tp"] = float(row.get("tp", 0) or 0)
            row["qty"] = int(row.get("qty", 0) or 0)
            by_sym[row["symbol"]].append(row)
    return by_sym


def _estimate_charges(buy_turnover: float, sell_turnover: float, executed_orders: int) -> dict[str, float]:
    turnover = buy_turnover + sell_turnover
    brokerage = executed_orders * 20.0
    stt = sell_turnover * 0.00025
    exchange_txn = turnover * 0.0000345
    sebi = turnover / 1e7 * 10.0
    stamp = buy_turnover * 0.00003
    gst = 0.18 * (brokerage + exchange_txn)
    total = brokerage + stt + exchange_txn + sebi + stamp + gst
    return {
        "brokerage": brokerage,
        "stt": stt,
        "exchange_txn": exchange_txn,
        "sebi": sebi,
        "stamp": stamp,
        "gst": gst,
        "total": total,
        "turnover": turnover,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], summary_rows: list[tuple[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        if rows:
            fieldnames = ["date", "symbol", "result", "pnl", "source"]
        else:
            fieldnames = ["date", "symbol", "result", "pnl", "source"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            if isinstance(out.get("date"), date):
                out["date"] = out["date"].isoformat()
            writer.writerow(out)
        writer.writerow({})
        writer.writerow({"date": "summary", "symbol": "metric", "result": "value", "pnl": "", "source": ""})
        for key, value in summary_rows:
            writer.writerow({"date": "summary", "symbol": key, "result": value, "pnl": "", "source": ""})


def _collect_historical_realized(day: date) -> tuple[list[dict], float, float, int, list[str]]:
    """
    Reconstruct realized trades for one historical day from the live log.

    Returns:
      trades, buy_turnover, sell_turnover, executed_orders, notes
    """
    log_path = LIVE_DIR / f"live_{day:%Y-%m-%d}.log"
    if not log_path.exists():
        return [], 0.0, 0.0, 0, [f"missing log for {day:%Y-%m-%d}"]

    signals = _load_signal_rows(day)
    lines = log_path.read_text().splitlines()

    entry_order_pending: dict[str, dict] = {}
    order_kind: dict[str, dict] = {}
    active: dict[str, dict] = {}
    trades: list[dict] = []
    buy_turnover = 0.0
    sell_turnover = 0.0
    executed_orders = 0
    notes: list[str] = []

    re_entry_order = re.compile(r"ORDER\s+(\S+)\s+(BUY|SELL)\s+qty=(\d+)\s+limit=([0-9.]+)")
    re_entry_ok = re.compile(r"ORDER OK\s+(\S+)\s+order_id=(\S+)")
    re_sl_ok = re.compile(r"SL ORDER OK\s+(\S+)\s+order_id=(\S+)")
    re_tp_ok = re.compile(r"TP ORDER OK\s+(\S+)\s+order_id=(\S+)")
    re_exit_ok = re.compile(r"EXIT ORDER OK\s+(\S+)\s+order_id=(\S+)")
    re_order_filled = re.compile(r"ORDER FILLED\s+(\S+)\s+order_id=(\S+)\s+qty=(\d+)\s+avg=([0-9.]+)")
    re_order_partial = re.compile(r"ORDER PARTIAL\s+(\S+)\s+filled=(\d+)")
    re_order_expiry = re.compile(r"ORDER EXPIRY\s+(\S+)\s+\S+\s+order_id=(\S+)\s+elapsed=.*?filled=(\d+)")
    re_broker_exit = re.compile(r"BROKER EXIT P&L\s+(\S+)\s+(SL|TP)\s+₹([+-]?\d+(?:\.\d+)?)")
    re_shutdown_exit = re.compile(r"SHUTDOWN EXIT P&L\s+(\S+)\s+(SL|TP)\s+₹([+-]?\d+(?:\.\d+)?)")
    re_bracket = re.compile(r"BRACKET\s+(\S+)\s+\S+\s+(SL|TP) hit .*?P&L ₹([+-]?\d+(?:\.\d+)?)")

    for line in lines:
        m = re_entry_order.search(line)
        if m:
            sym, side, qty, limit = m.groups()
            entry_order_pending[sym] = {
                "side": side,
                "qty": int(qty),
                "limit": float(limit),
            }
            continue

        for regex, kind in ((re_entry_ok, "entry"), (re_sl_ok, "sl"), (re_tp_ok, "tp"), (re_exit_ok, "exit")):
            m = regex.search(line)
            if m:
                sym, oid = m.groups()
                if kind == "entry":
                    meta = entry_order_pending.get(sym, {})
                    order_kind[oid] = {"kind": "entry", "sym": sym, **meta}
                else:
                    order_kind[oid] = {"kind": kind, "sym": sym}
                break
        else:
            pass

        m = re_order_filled.search(line)
        if m:
            sym, oid, qty_s, avg_s = m.groups()
            qty = int(qty_s)
            avg = float(avg_s)
            meta = order_kind.get(oid, {"kind": "unknown", "sym": sym})
            kind = meta["kind"]
            if kind == "entry":
                executed_orders += 1
                if meta.get("side") == "BUY":
                    buy_turnover += qty * avg
                    direction = "LONG"
                else:
                    sell_turnover += qty * avg
                    direction = "SHORT"
                sig = signals[sym].popleft() if signals[sym] else {"sl": 0.0, "tp": 0.0, "variant": "?"}
                active[sym] = {
                    "direction": direction,
                    "entry_price": avg,
                    "qty": qty,
                    "sl": float(sig.get("sl", 0) or 0),
                    "tp": float(sig.get("tp", 0) or 0),
                    "variant": sig.get("variant", "?"),
                    "source": "entry_fill",
                }
            elif kind == "exit":
                executed_orders += 1
                if meta.get("sym") in active:
                    if active[meta["sym"]]["direction"] == "LONG":
                        sell_turnover += qty * avg
                    else:
                        buy_turnover += qty * avg
            continue

        m = re_order_partial.search(line)
        if m:
            sym, qty_s = m.groups()
            qty = int(qty_s)
            meta = entry_order_pending.get(sym)
            if meta and qty > 0:
                executed_orders += 1
                avg = meta["limit"]
                if meta.get("side") == "BUY":
                    buy_turnover += qty * avg
                    direction = "LONG"
                else:
                    sell_turnover += qty * avg
                    direction = "SHORT"
                sig = signals[sym].popleft() if signals[sym] else {"sl": 0.0, "tp": 0.0, "variant": "?"}
                active[sym] = {
                    "direction": direction,
                    "entry_price": avg,
                    "qty": qty,
                    "sl": float(sig.get("sl", 0) or 0),
                    "tp": float(sig.get("tp", 0) or 0),
                    "variant": sig.get("variant", "?"),
                    "source": "partial_entry",
                }
            continue

        m = re_order_expiry.search(line)
        if m:
            sym, oid, filled_s = m.groups()
            if int(filled_s) == 0 and sym not in active:
                notes.append(f"{sym}: entry expired unfilled")
            continue

        for regex, source in ((re_broker_exit, "broker_exit"), (re_shutdown_exit, "shutdown_exit"), (re_bracket, "bracket")):
            m = regex.search(line)
            if m:
                sym, result, pnl_s = m.groups()
                pnl = float(pnl_s)
                pos = active.get(sym)
                if source == "bracket" and not pos:
                    # likely phantom/stale trade; skip
                    notes.append(f"{sym}: skipped phantom {source} exit")
                    break
                if pos:
                    exit_price = pos["tp"] if result == "TP" else pos["sl"]
                    qty = int(pos["qty"])
                    if pos["direction"] == "LONG":
                        sell_turnover += qty * exit_price
                    else:
                        buy_turnover += qty * exit_price
                    executed_orders += 1
                    trades.append({
                        "date": day,
                        "symbol": sym,
                        "result": result,
                        "pnl": pnl,
                        "source": source,
                    })
                    del active[sym]
                else:
                    # fallback: trust P&L line but mark note
                    trades.append({
                        "date": day,
                        "symbol": sym,
                        "result": result,
                        "pnl": pnl,
                        "source": f"{source}_unmatched",
                    })
                    notes.append(f"{sym}: unmatched {source} P&L included without turnover reconstruction")
                break

    return trades, buy_turnover, sell_turnover, executed_orders, notes


def _collect_today_broker(today: date) -> tuple[list[dict], float, float, int, float]:
    client = AngelOneClient().connect()
    conn = client._conn

    pos_resp = conn.position()
    positions = pos_resp.get("data") or []
    intraday = [p for p in positions if p.get("producttype") == "INTRADAY"]

    trade_resp = conn.tradeBook()
    trades = trade_resp.get("data") or []

    order_resp = conn.orderBook()
    orders = order_resp.get("data") or []
    completed_orders = [o for o in orders if str(o.get("orderstatus", "")).lower() in ("complete", "filled")]

    by_symbol = []
    realized_total = 0.0
    unrealized_total = 0.0
    for p in intraday:
        realised = float(p.get("realised", 0) or 0)
        unrealised = float(p.get("unrealised", 0) or 0)
        realized_total += realised
        unrealized_total += unrealised
        by_symbol.append({
            "date": today,
            "symbol": p.get("tradingsymbol", ""),
            "result": "broker",
            "pnl": realised,
            "source": "position_realised",
        })

    buy_turnover = 0.0
    sell_turnover = 0.0
    for t in trades:
        qty = int(t.get("fillsize", 0) or 0)
        px = float(t.get("fillprice", 0) or 0)
        tv = qty * px
        if t.get("transactiontype") == "BUY":
            buy_turnover += tv
        else:
            sell_turnover += tv

    return by_symbol, buy_turnover, sell_turnover, len(completed_orders), unrealized_total


def main() -> None:
    parser = argparse.ArgumentParser(description="Angel One aligned P&L report")
    parser.add_argument("--date", type=str, help="Single day report (YYYY-MM-DD)")
    parser.add_argument("--start-date", type=str, help="YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, help="YYYY-MM-DD")
    parser.add_argument("--csv", type=str, help="Write the report rows/summary to a CSV file")
    args = parser.parse_args()

    today = datetime.now().date()
    if args.date:
        start = _parse_date(args.date)
        end = start
    else:
        start = _parse_date(args.start_date) if args.start_date else today
        end = _parse_date(args.end_date) if args.end_date else today
    if start > end:
        raise SystemExit("start-date must be <= end-date")

    print("=" * 104)
    print(f"  ANGEL ONE BROKER P&L REPORT — {start:%Y-%m-%d} to {end:%Y-%m-%d}")
    print("=" * 104)

    all_trades: list[dict] = []
    buy_turnover = 0.0
    sell_turnover = 0.0
    executed_orders = 0
    unrealized_total = 0.0
    notes: list[str] = []

    current = start
    while current <= end:
        if current == today:
            trades, buy_tv, sell_tv, exec_orders, unreal = _collect_today_broker(current)
            all_trades.extend(trades)
            buy_turnover += buy_tv
            sell_turnover += sell_tv
            executed_orders += exec_orders
            unrealized_total += unreal
        else:
            trades, buy_tv, sell_tv, exec_orders, day_notes = _collect_historical_realized(current)
            all_trades.extend(trades)
            buy_turnover += buy_tv
            sell_turnover += sell_tv
            executed_orders += exec_orders
            notes.extend(day_notes)
        current = current + timedelta(days=1)

    # Day-by-day detail
    print("\nRealized trades / broker rows:")
    print(f"  {'Date':<12} {'Symbol':<16} {'Result':<8} {'P&L':>12} {'Source':<18}")
    print("  " + "-" * 76)
    for row in sorted(all_trades, key=lambda r: (r["date"], r["symbol"], r["source"])):
        print(
            f"  {row['date']:%Y-%m-%d} {row['symbol']:<16} {row['result']:<8} "
            f"{row['pnl']:>12.2f} {row['source']:<18}"
        )

    daily = defaultdict(float)
    for row in all_trades:
        daily[row["date"]] += row["pnl"]

    gross_realized = sum(row["pnl"] for row in all_trades)
    charges = _estimate_charges(buy_turnover, sell_turnover, executed_orders)

    print("\nDaily totals:")
    print("  " + "-" * 32)
    for day in sorted(daily):
        print(f"  {day:%Y-%m-%d}  {_money(daily[day])}")

    print("\nTurnover / orders:")
    print(f"  Buy turnover:       {_money(buy_turnover)}")
    print(f"  Sell turnover:      {_money(sell_turnover)}")
    print(f"  Total turnover:     {_money(charges['turnover'])}")
    print(f"  Executed orders:    {executed_orders}")

    print("\nEstimated charges:")
    print(f"  Brokerage:          {_money(charges['brokerage'])}")
    print(f"  STT:                {_money(charges['stt'])}")
    print(f"  Exchange txn:       {_money(charges['exchange_txn'])}")
    print(f"  SEBI:               {_money(charges['sebi'])}")
    print(f"  Stamp duty:         {_money(charges['stamp'])}")
    print(f"  GST:                {_money(charges['gst'])}")
    print(f"  Total charges:      {_money(charges['total'])}")

    print("\nTotals:")
    print(f"  Gross realized P&L: {_money(gross_realized)}")
    print(f"  Unrealized P&L:     {_money(unrealized_total)}")
    print(f"  Net est. realized:  {_money(gross_realized - charges['total'])}")
    print(f"  Net est. total:     {_money(gross_realized + unrealized_total - charges['total'])}")

    print("\nNotes:")
    print("  - Current day uses Angel One position()/tradeBook()/orderBook() directly.")
    print("  - Prior days are reconstructed from archived live logs in a broker-aligned way.")
    print("  - Charges are still estimated because Angel One APIs here do not expose a charge breakup.")
    if notes:
        for note in notes:
            print(f"  - {note}")

    if args.csv:
        summary_rows = [
            ("start_date", start.isoformat()),
            ("end_date", end.isoformat()),
            ("gross_realized_pnl", f"{gross_realized:.2f}"),
            ("unrealized_pnl", f"{unrealized_total:.2f}"),
            ("buy_turnover", f"{buy_turnover:.2f}"),
            ("sell_turnover", f"{sell_turnover:.2f}"),
            ("executed_orders", executed_orders),
            ("brokerage", f"{charges['brokerage']:.2f}"),
            ("stt", f"{charges['stt']:.2f}"),
            ("exchange_txn", f"{charges['exchange_txn']:.2f}"),
            ("sebi", f"{charges['sebi']:.2f}"),
            ("stamp", f"{charges['stamp']:.2f}"),
            ("gst", f"{charges['gst']:.2f}"),
            ("total_charges", f"{charges['total']:.2f}"),
            ("net_est_realized", f"{gross_realized - charges['total']:.2f}"),
            ("net_est_total", f"{gross_realized + unrealized_total - charges['total']:.2f}"),
        ]
        _write_csv(Path(args.csv), all_trades, summary_rows)
        print(f"\nCSV written: {Path(args.csv).resolve()}")


if __name__ == "__main__":
    main()

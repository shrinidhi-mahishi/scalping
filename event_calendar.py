"""
Event Calendar Filter — skips stocks with corporate events on trading day.

Fetches from NSE India:
  - Board meetings (quarterly results, AGMs)
  - Corporate actions (ex-dividend, bonus, splits, buybacks)

Results are cached in data/event_cache.json (refreshed every 12 hours).
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests

log = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "data"
CACHE_FILE = CACHE_DIR / "event_cache.json"
CACHE_TTL_HOURS = 12

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-board-meetings",
}

_NSE_BASE = "https://www.nseindia.com"

EVENT_REASONS = {
    "results": ["Financial Results", "Quarterly Results", "Annual Results",
                "Un-Audited Financial Results", "Audited Financial Results",
                "Half Yearly Results"],
    "agm": ["AGM", "Annual General Meeting", "EGM", "Extra Ordinary General Meeting"],
    "dividend": ["Dividend", "Interim Dividend", "Final Dividend"],
    "split": ["Stock Split", "Face Value Split", "Sub-Division"],
    "bonus": ["Bonus", "Bonus Issue"],
    "buyback": ["Buyback", "Buy Back"],
    "rights": ["Rights Issue", "Rights"],
}


def _nse_session() -> requests.Session:
    """Create an NSE session with proper cookies."""
    s = requests.Session()
    s.headers.update(_HEADERS)
    try:
        s.get(_NSE_BASE, timeout=10)
        time.sleep(0.5)
    except Exception as e:
        log.warning("NSE homepage fetch failed (cookie init): %s", e)
    return s


def _fetch_board_meetings(session: requests.Session, from_date: str, to_date: str) -> list[dict]:
    url = f"{_NSE_BASE}/api/corporate-board-meetings"
    params = {"index": "equities", "from_date": from_date, "to_date": to_date}
    try:
        resp = session.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []
    except Exception as e:
        log.warning("Board meetings fetch failed: %s", e)
        return []


def _fetch_corporate_actions(session: requests.Session, from_date: str, to_date: str) -> list[dict]:
    url = f"{_NSE_BASE}/api/corporates-corporateActions"
    params = {"index": "equities", "from_date": from_date, "to_date": to_date}
    try:
        resp = session.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []
    except Exception as e:
        log.warning("Corporate actions fetch failed: %s", e)
        return []


def _classify_event(purpose: str) -> Optional[str]:
    """Map a raw NSE purpose string to an event category."""
    purpose_lower = purpose.lower()
    for category, keywords in EVENT_REASONS.items():
        for kw in keywords:
            if kw.lower() in purpose_lower:
                return category
    return None


def _parse_nse_date(date_str: str) -> Optional[str]:
    """Parse various NSE date formats to YYYY-MM-DD."""
    for fmt in ("%d-%b-%Y", "%d-%m-%Y", "%d %b %Y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(date_str.strip(), fmt).strftime("%Y-%m-%d")
        except (ValueError, AttributeError):
            continue
    return None


def _build_event_map(stocks: list[str], days_ahead: int = 3) -> dict[str, list[dict]]:
    """
    Fetch events for the given stocks from NSE.
    Returns {date_str: [{sym, category, detail}, ...]}.
    """
    today = datetime.now().date()
    from_date = today.strftime("%d-%m-%Y")
    to_date = (today + timedelta(days=days_ahead)).strftime("%d-%m-%Y")
    stock_set = {s.upper() for s in stocks}

    event_map: dict[str, list[dict]] = {}
    session = _nse_session()

    bm = _fetch_board_meetings(session, from_date, to_date)
    for item in bm:
        sym = (item.get("bm_symbol") or item.get("symbol") or "").upper().strip()
        if sym not in stock_set:
            continue
        raw_date = item.get("bm_date") or item.get("meeting_date") or ""
        purpose = item.get("bm_purpose") or item.get("purpose") or ""
        cat = _classify_event(purpose)
        if not cat:
            cat = "board_meeting"
        parsed = _parse_nse_date(raw_date)
        if parsed:
            event_map.setdefault(parsed, []).append({
                "sym": sym, "category": cat,
                "detail": purpose[:80],
            })

    time.sleep(1.0)

    ca = _fetch_corporate_actions(session, from_date, to_date)
    for item in ca:
        sym = (item.get("symbol") or "").upper().strip()
        if sym not in stock_set:
            continue
        raw_date = item.get("exDate") or item.get("recordDate") or ""
        subject = item.get("subject") or ""
        cat = _classify_event(subject)
        if not cat:
            continue
        parsed = _parse_nse_date(raw_date)
        if parsed:
            already = any(e["sym"] == sym and e["category"] == cat
                          for e in event_map.get(parsed, []))
            if not already:
                event_map.setdefault(parsed, []).append({
                    "sym": sym, "category": cat,
                    "detail": subject[:80],
                })

    return event_map


def _load_cache() -> Optional[dict]:
    """Load cached events if fresh."""
    if not CACHE_FILE.exists():
        return None
    try:
        raw = json.loads(CACHE_FILE.read_text())
        cached_at = datetime.fromisoformat(raw["cached_at"])
        if datetime.now() - cached_at > timedelta(hours=CACHE_TTL_HOURS):
            return None
        return raw["events"]
    except Exception:
        return None


def _save_cache(events: dict) -> None:
    """Save events to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"cached_at": datetime.now().isoformat(), "events": events}
    CACHE_FILE.write_text(json.dumps(payload, indent=2))


def get_excluded_stocks(
    stocks: list[str],
    target_date: Optional[datetime] = None,
    force_refresh: bool = False,
) -> dict[str, str]:
    """
    Returns {symbol: reason} for stocks that should be skipped today.

    Checks for:
      - Quarterly results / board meetings
      - Ex-dividend / ex-bonus / ex-split dates
      - AGM / EGM dates
    """
    if target_date is None:
        target_date = datetime.now()
    date_key = target_date.strftime("%Y-%m-%d")

    events = None
    if not force_refresh:
        events = _load_cache()

    if events is None:
        log.info("Fetching event calendar from NSE...")
        try:
            events = _build_event_map(stocks)
            _save_cache(events)
            log.info("Event calendar cached (%d dates with events)", len(events))
        except Exception as e:
            log.warning("Event calendar fetch failed: %s — no stocks excluded", e)
            return {}

    excluded: dict[str, str] = {}
    for ev in events.get(date_key, []):
        sym = ev["sym"]
        cat = ev["category"]
        detail = ev["detail"]

        if cat in ("results", "agm", "dividend", "split", "bonus", "buyback", "rights"):
            reason = f"{cat.upper()}: {detail}"
            excluded[sym] = reason

    return excluded


if __name__ == "__main__":
    import sys
    test_stocks = [
        "BAJAJ-AUTO", "POWERGRID", "TATACONSUM", "INDIGO", "BAJFINANCE",
        "HCLTECH", "JIOFIN", "EICHERMOT", "NESTLEIND", "BHARTIARTL",
    ]
    print(f"Checking events for: {', '.join(test_stocks)}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    print()

    exc = get_excluded_stocks(test_stocks, force_refresh=True)
    if exc:
        print("EXCLUDED STOCKS:")
        for sym, reason in exc.items():
            print(f"  {sym:<14} — {reason}")
    else:
        print("No events found — all stocks cleared for trading.")

    print("\nFull event cache:")
    if CACHE_FILE.exists():
        cache = json.loads(CACHE_FILE.read_text())
        events = cache.get("events", {})
        if events:
            for date_key in sorted(events.keys()):
                for ev in events[date_key]:
                    print(f"  {date_key}  {ev['sym']:<14}  {ev['category']:<12}  {ev['detail']}")
        else:
            print("  (empty)")

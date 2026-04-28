"""
Compare baseline Prearmed PDL vs a Nifty regime gate.

Gate definition on the same 3-minute bar:
  - LONG allowed only if Nifty 50 close > Nifty day open
  - SHORT allowed only if Nifty 50 close < Nifty day open
  - Equal-to-open is treated as neutral and blocks both

Uses the current top-10 basket and current live strategy constants.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(".").resolve()))
from fetch_data import AngelOneClient, load_csv
from live_signals import (
    STOCKS,
    RR,
    RISK_PCT,
    LEV_CAP,
    PDL_PREARM_BUFFER_ATR,
    PDL_SL_MULT,
    MAX_SL_PER_DAY,
    ENTRY_AM,
    ENTRY_PM,
    COOLDOWN,
    compute_indicators,
)


CASH = 15_000.0
TODAY = pd.Timestamp.now().normalize()
PERIODS = [("30D", 30), ("90D", 90)]
NIFTY_SYMBOL = "NIFTY50"
NIFTY_TOKEN = "99926000"
NIFTY_CACHE = Path("data") / "_nifty50_3min_cache.csv"
_F = str.maketrans({"&": ""})


def csv_path(sym: str) -> Path:
    fname = sym.translate(_F)
    p = Path("data") / f"{fname}_3min.csv"
    return p if p.exists() else Path("data") / f"{fname.lower()}_3min.csv"


def csv_path_1min(sym: str) -> Path:
    fname = sym.translate(_F)
    return Path("data") / f"{fname}_1min.csv"


def resample_3min(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("3min").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    ).dropna(subset=["open"])


def load_symbol_df(sym: str) -> pd.DataFrame | None:
    p1 = csv_path_1min(sym)
    if p1.exists():
        return resample_3min(load_csv(str(p1)))
    p3 = csv_path(sym)
    if p3.exists():
        return load_csv(str(p3))
    return None


def load_nifty_df(days: int) -> pd.DataFrame:
    AngelOneClient.SYMBOL_TOKENS[NIFTY_SYMBOL] = NIFTY_TOKEN
    client = AngelOneClient()
    try:
        client.connect()
        df = client.fetch_history(NIFTY_SYMBOL, days=days, interval="THREE_MINUTE")
        NIFTY_CACHE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(NIFTY_CACHE)
        return df
    except Exception:
        if NIFTY_CACHE.exists():
            return load_csv(str(NIFTY_CACHE))
        raise


def build_day_slices(df: pd.DataFrame, start: pd.Timestamp):
    try:
        ind = compute_indicators(df)
        ind = ind[ind.index >= start]
        if len(ind) < 50:
            return None
        days = sorted(ind.index.normalize().unique())
        out = []
        for i in range(1, len(days)):
            td, pd_ = days[i], days[i - 1]
            pr = ind[ind.index.normalize() == pd_]
            tr = ind[ind.index.normalize() == td]
            if pr.empty or tr.empty:
                continue
            out.append((td, float(pr["high"].max()), float(pr["low"].min()), tr))
        return out
    except Exception:
        return None


def build_nifty_regime_lookup(df: pd.DataFrame, start: pd.Timestamp) -> dict[pd.Timestamp, str]:
    df = df[df.index >= start].copy()
    if df.empty:
        return {}
    day_open = df.groupby(df.index.normalize())["open"].transform("first")
    regime = pd.Series("neutral", index=df.index, dtype="object")
    regime[df["close"] > day_open] = "bullish"
    regime[df["close"] < day_open] = "bearish"
    return regime.to_dict()


def in_entry_window(t) -> bool:
    if ENTRY_AM and ENTRY_AM[0] <= t <= ENTRY_AM[1]:
        return True
    if ENTRY_PM and ENTRY_PM[0] <= t <= ENTRY_PM[1]:
        return True
    return False


def simulate(day_data, sym: str, regime_lookup: dict[pd.Timestamp, str], use_gate: bool):
    trades = []
    blocked_long = 0
    blocked_short = 0

    for today, pdl_h, pdl_l, today_rows in day_data:
        pdl_dirs, cd_until, pos, activate_idx, dsl = set(), None, None, None, 0

        for i in range(len(today_rows)):
            row = today_rows.iloc[i]
            ts = today_rows.index[i]
            t = ts.time()
            c = float(row["close"])

            if pos is not None:
                if activate_idx is not None and i >= activate_idx:
                    hit, pv = None, 0.0
                    if pos["d"] == "LONG":
                        if row["low"] <= pos["sl"]:
                            hit, pv = "SL", (pos["sl"] - pos["e"]) * pos["q"]
                        elif row["high"] >= pos["tp"]:
                            hit, pv = "TP", (pos["tp"] - pos["e"]) * pos["q"]
                    else:
                        if row["high"] >= pos["sl"]:
                            hit, pv = "SL", (pos["e"] - pos["sl"]) * pos["q"]
                        elif row["low"] <= pos["tp"]:
                            hit, pv = "TP", (pos["e"] - pos["tp"]) * pos["q"]
                    if hit:
                        trades.append({**pos, "pnl": pv, "res": hit, "et": ts})
                        if hit == "SL":
                            dsl += 1
                        pos, activate_idx = None, None
                continue

            if not in_entry_window(t) or (cd_until and ts < cd_until) or dsl >= MAX_SL_PER_DAY:
                continue

            atr = float(row["atr"])
            prev_close = float(today_rows.iloc[i - 1]["close"]) if i > 0 else None
            if prev_close is None or pd.isna(atr) or atr <= 0:
                continue

            buffer = atr * PDL_PREARM_BUFFER_ATR
            tick_tol = atr * 0.05
            sig, entry = None, None

            if (
                "LONG" not in pdl_dirs
                and prev_close <= pdl_h + tick_tol
                and row["high"] >= pdl_h + buffer
                and c >= pdl_h
            ):
                sig, entry = "LONG", pdl_h + buffer
            elif (
                "SHORT" not in pdl_dirs
                and prev_close >= pdl_l - tick_tol
                and row["low"] <= pdl_l - buffer
                and c <= pdl_l
            ):
                sig, entry = "SHORT", pdl_l - buffer

            if sig and use_gate:
                regime = regime_lookup.get(ts, "neutral")
                if sig == "LONG" and regime != "bullish":
                    blocked_long += 1
                    continue
                if sig == "SHORT" and regime != "bearish":
                    blocked_short += 1
                    continue

            if sig:
                sd = atr * PDL_SL_MULT
                rq = int(CASH * RISK_PCT / sd)
                mq = int(CASH * LEV_CAP / entry)
                q = max(min(rq, mq), 1)
                slp = entry - sd if sig == "LONG" else entry + sd
                tpp = entry + sd * RR if sig == "LONG" else entry - sd * RR
                pos = {
                    "sym": sym,
                    "d": sig,
                    "trig": "PDL",
                    "e": entry,
                    "sl": slp,
                    "tp": tpp,
                    "atr": atr,
                    "q": q,
                    "date": today,
                    "etime": ts,
                }
                cd_until = ts + COOLDOWN
                activate_idx = i + 1
                pdl_dirs.add(sig)

    return trades, blocked_long, blocked_short


def analyze(trades):
    if not trades:
        return {
            "n": 0,
            "wins": 0,
            "wr": 0.0,
            "pnl": 0.0,
            "tp": 0,
            "sl": 0,
            "dd": 0.0,
            "avg": 0.0,
            "pf": 0.0,
        }

    tdf = pd.DataFrame(trades)
    n = len(tdf)
    pnl = float(tdf["pnl"].sum())
    wins = int((tdf["pnl"] > 0).sum())
    tp = int((tdf["res"] == "TP").sum())
    sl = int((tdf["res"] == "SL").sum())
    cumul = tdf["pnl"].cumsum()
    dd = float((cumul - cumul.cummax()).min())
    gw = float(tdf.loc[tdf["pnl"] > 0, "pnl"].sum())
    gl = abs(float(tdf.loc[tdf["pnl"] <= 0, "pnl"].sum()))
    pf = gw / gl if gl > 0 else (9.99 if gw > 0 else 0.0)
    return {
        "n": n,
        "wins": wins,
        "wr": wins / n * 100.0,
        "pnl": pnl,
        "tp": tp,
        "sl": sl,
        "dd": dd,
        "avg": pnl / n,
        "pf": min(pf, 9.99),
    }


def run():
    max_days = max(days for _, days in PERIODS) + 5
    nifty_df = load_nifty_df(max_days)

    stock_frames = {}
    for sym, _ in STOCKS:
        df = load_symbol_df(sym)
        if df is not None:
            stock_frames[sym] = df

    print("=" * 118)
    print("  NIFTY REGIME FILTER TEST — current top basket")
    print(f"  Window: {ENTRY_AM[0]:%H:%M}-{ENTRY_AM[1]:%H:%M}  |  RR={RR}  |  Capital=₹{CASH:,.0f}")
    print("  Gate: LONG only if Nifty 3-min close > Nifty day open; SHORT only if close < day open")
    print(f"  Stocks: {', '.join(s for s, _ in STOCKS)}")
    print("=" * 118)

    results = []
    for label, days in PERIODS:
        start = TODAY - pd.Timedelta(days=days)
        regime_lookup = build_nifty_regime_lookup(nifty_df, start)

        baseline_trades = []
        gated_trades = []
        blocked_long = 0
        blocked_short = 0

        for sym, _ in STOCKS:
            df = stock_frames.get(sym)
            if df is None:
                continue
            day_data = build_day_slices(df, start)
            if not day_data:
                continue

            t_base, _, _ = simulate(day_data, sym, regime_lookup, use_gate=False)
            t_gate, bl, bs = simulate(day_data, sym, regime_lookup, use_gate=True)
            baseline_trades.extend(t_base)
            gated_trades.extend(t_gate)
            blocked_long += bl
            blocked_short += bs

        base = analyze(baseline_trades)
        gate = analyze(gated_trades)
        results.append(
            {
                "period": label,
                "baseline": base,
                "gated": gate,
                "blocked_long": blocked_long,
                "blocked_short": blocked_short,
            }
        )

    print()
    print(
        f"  {'Period':<6} │ {'Variant':<10} │ {'Tr':>4} {'WR%':>6} {'TP':>4} {'SL':>4} │ "
        f"{'P&L':>9} {'Avg':>7} {'MaxDD':>8} {'PF':>5} │ {'Blk L':>5} {'Blk S':>5}"
    )
    print("  " + "─" * 111)

    for r in results:
        base = r["baseline"]
        gate = r["gated"]
        print(
            f"  {r['period']:<6} │ {'Baseline':<10} │ {base['n']:>4} {base['wr']:>5.1f}% {base['tp']:>4} {base['sl']:>4} │ "
            f"₹{base['pnl']:>7,.0f} ₹{base['avg']:>5,.0f} ₹{base['dd']:>6,.0f} {base['pf']:>5.2f} │ {0:>5} {0:>5}"
        )
        print(
            f"  {r['period']:<6} │ {'NiftyGate':<10} │ {gate['n']:>4} {gate['wr']:>5.1f}% {gate['tp']:>4} {gate['sl']:>4} │ "
            f"₹{gate['pnl']:>7,.0f} ₹{gate['avg']:>5,.0f} ₹{gate['dd']:>6,.0f} {gate['pf']:>5.2f} │ "
            f"{r['blocked_long']:>5} {r['blocked_short']:>5}"
        )
        delta_pnl = gate["pnl"] - base["pnl"]
        delta_wr = gate["wr"] - base["wr"]
        print(
            f"  {'':<6} │ {'Delta':<10} │ {gate['n'] - base['n']:>4} {delta_wr:>+5.1f}%{' ':>8} │ "
            f"₹{delta_pnl:>+7,.0f}{' ':>23}"
        )
        print("  " + "─" * 111)


if __name__ == "__main__":
    run()

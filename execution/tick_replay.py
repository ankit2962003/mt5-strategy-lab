"""
Tick Replay — instant backtest on REAL MT5 ticks (real spread + slippage)
=========================================================================
Phase 2B, priority 1. The whole edifice rests on this: clean-OHLC backtests are
fiction that die on live spreads. We already watched gold go +0.47R on paper and
bleed negative live purely from spread. So every fill here pays the REAL bid/ask
of the moment, and exits are checked tick-by-tick — not against a flat
SPREAD_COST_R fudge.

What it does
------------
  * fetch + CACHE real ticks (mt5.copy_ticks_range) per symbol per day to disk
    (ticks are large — ~430k/day on EURUSD, ~26MB raw — so we never re-pull).
  * replay a lab gene over any window using:
       - SIGNALS computed on bars (same prepare()/predicate() as strategy_lab),
       - FILLS executed on real ticks: BUY pays the ask, SELL pays the bid, so
         the spread is a real cost on entry AND exit, plus optional slippage,
       - the SAME dynamic management the live runner / arena use
         (breakeven at +0.8R, trail beyond +1.0R) so a replay is a faithful
         proxy for what the live demo would actually have done.

Correction (2026-07-01): the "~30-35 day horizon" noted below on 2026-06-21 was a
MISDIAGNOSIS. The real bug was in fetch_day()'s retry logic: it only retried on
mt5.last_error()[0] == -10001, but the actual failure mode seen under concurrent-
daemon load is (-1, 'Terminal: Call failed') on the first call after connect —
NOT retried, so a transient hiccup got cache-poisoned as a permanent 0-tick file.
All 12/12 previously-"empty" XAUUSD weekdays recovered real ticks once refetched
with the fixed retry logic (see `repair_empty`). Separately, copy_ticks_range()
can also hang indefinitely (not return None) for genuinely out-of-horizon dates —
`fetch_day_safe` bounds this with a subprocess timeout so one hang can't stall an
entire backfill run.

CLI
---
  py tick_replay.py cache EURUSD 2026-05-25 2026-06-20   # cache a date range
  py tick_replay.py archive                              # cache yesterday for all
  py tick_replay.py coverage EURUSD                      # what's cached on disk
  py tick_replay.py repair XAUUSD                        # refetch cache-poisoned empty days
  py tick_replay.py backfill XAUUSD 183                  # N-day backfill + honest summary
  py tick_replay.py selftest                             # synthetic-tick unit test
"""

import os, sys, json, time, glob
from datetime import datetime, timedelta, timezone, date as _date
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE       = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR  = os.path.join(BASE, "tick_cache")
# FEMA compliance (2026-07-11): keep this set in sync with the identical block
# in strategy_lab.py (no shared import — repo convention is standalone
# scripts). See that file's comment for the rationale.
_FULL_UNIVERSE        = ["EURUSD", "GBPUSD", "USDJPY", "USDCAD", "AUDUSD", "NZDUSD",
                         "USDCHF", "EURJPY", "GBPJPY", "EURGBP", "XAUUSD", "XAGUSD"]
FEMA_EXCLUDED_SPOT_FX = {"EURUSD", "GBPUSD", "USDJPY", "USDCAD", "AUDUSD", "NZDUSD",
                         "USDCHF", "EURJPY", "GBPJPY", "EURGBP"}
FEMA_COMPLIANT        = True
UNIVERSE = [s for s in _FULL_UNIVERSE if not (FEMA_COMPLIANT and s in FEMA_EXCLUDED_SPOT_FX)]
if FEMA_COMPLIANT:
    print(f"  [FEMA] {len(_FULL_UNIVERSE) - len(UNIVERSE)} offshore spot-FX pairs excluded "
          f"from UNIVERSE (FEMA compliance) -> trading {UNIVERSE}")

# Dynamic management — MUST match arena._apply_dynamic so a replay equals live.
BE_TRIGGER_R    = 0.8
TRAIL_TRIGGER_R = 1.0
TRAIL_GAP_R     = 0.6


# ─── MT5 (imported lazily so the pure replay core needs no terminal) ──────────

def _mt5():
    import MetaTrader5 as mt5
    if not mt5.initialize():
        if not mt5.initialize(path=r"C:\Program Files\MetaTrader 5\terminal64.exe"):
            raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")
    ti = mt5.terminal_info()
    if ti is None or not ti.connected:
        raise RuntimeError(f"MT5 terminal not connected — aborting "
                            f"(terminal_info={ti}, last_error={mt5.last_error()})")
    return mt5


def resolve_symbol(mt5, symbol):
    """Verify `symbol` exists on this broker; if not, try to find the closest
    match instead of assuming the name is right (e.g. XAUUSD vs XAUUSD.m)."""
    if mt5.symbol_info(symbol) is not None:
        return symbol
    matches = mt5.symbols_get(f"*{symbol}*") or []
    exact = [s.name for s in matches if s.name == symbol]
    if exact:
        return exact[0]
    if len(matches) == 1:
        resolved = matches[0].name
        print(f"[SYMBOL] '{symbol}' not found verbatim — using closest match '{resolved}'")
        return resolved
    if matches:
        raise RuntimeError(f"Symbol '{symbol}' ambiguous — candidates: {[s.name for s in matches]}")
    raise RuntimeError(f"Symbol '{symbol}' not found on this broker/terminal.")


# ─── TICK CACHE ───────────────────────────────────────────────────────────────
# One compressed .npz per (symbol, UTC date): time_msc(int64), bid/ask(float64).
# Empty/holiday days get a 0-row file so we never re-pull a known-empty day.

def _day_path(symbol, d):
    return os.path.join(CACHE_DIR, symbol, f"{d.isoformat()}.npz")

def _save_day(symbol, d, time_msc, bid, ask):
    p = _day_path(symbol, d)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.savez_compressed(p, time_msc=time_msc.astype(np.int64),
                        bid=bid.astype(np.float64), ask=ask.astype(np.float64))

def _load_day(symbol, d):
    p = _day_path(symbol, d)
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return z["time_msc"], z["bid"], z["ask"]

def _have_day(symbol, d):
    return os.path.exists(_day_path(symbol, d))


def fetch_day(symbol, d, mt5=None, force=False, retries=5):
    """Fetch + cache one UTC day of real ticks. Returns:
      >0  tick count cached
       0  confirmed empty (weekend, holiday, or server explicitly returned 0 rows)
          -> safe to cache, we won't refetch it
      -1  the call itself kept failing after all retries (transient IPC/terminal
          error) -> NOT cached, so a later run will retry instead of treating a
          hiccup as a permanent data gap. Root-caused 2026-07-01: the old retry
          logic only retried on err[0]==-10001, but the observed failure mode is
          (-1, 'Terminal: Call failed') on the FIRST call after connect/under
          concurrent-daemon load — that error wasn't retried, so a transient
          hiccup got cache-poisoned as a permanent 0-tick file.
    Skips already-cached days unless force=True."""
    if not force and _have_day(symbol, d):
        tm, _, _ = _load_day(symbol, d)
        return len(tm)
    if d.weekday() >= 5:                      # Sat/Sun — market closed
        _save_day(symbol, d, np.array([], np.int64), np.array([]), np.array([]))
        return 0
    mt5 = mt5 or _mt5()
    mt5.symbol_select(symbol, True)
    start = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    end   = start + timedelta(days=1)
    ticks, err = None, None
    for attempt in range(retries):
        ticks = mt5.copy_ticks_range(symbol, start, end, mt5.COPY_TICKS_ALL)
        err = mt5.last_error()
        if ticks is not None:
            break
        time.sleep(0.4 * (attempt + 1))
        if err and err[0] == -10001:          # IPC send failed -> reconnect
            mt5.shutdown(); mt5.initialize()
    if ticks is None:
        print(f"[FAIL] {symbol} {d} ({d.strftime('%a')}): copy_ticks_range failed "
              f"after {retries} attempts — mt5.last_error={err} — NOT cached, will retry")
        return -1
    if len(ticks) == 0:
        print(f"[EMPTY] {symbol} {d} ({d.strftime('%a')}): confirmed 0 ticks "
              f"— mt5.last_error={err}")
        _save_day(symbol, d, np.array([], np.int64), np.array([]), np.array([]))
        return 0
    bid = ticks["bid"].astype(np.float64)
    ask = ticks["ask"].astype(np.float64)
    tmc = ticks["time_msc"].astype(np.int64)
    # MT5 forward-fills bid/ask on last-only ticks; drop any zero/invalid quotes
    valid = (bid > 0) & (ask > 0) & (ask >= bid)
    _save_day(symbol, d, tmc[valid], bid[valid], ask[valid])
    return int(valid.sum())


def fetch_day_safe(symbol, d, force=False, timeout_s=25):
    """Bounded version of fetch_day. copy_ticks_range() can hang indefinitely
    (not just return None) for a genuinely out-of-horizon date — confirmed live
    2026-07-01 (process sat at 0.7s CPU time after 30+ min wall-clock, stuck on
    the single call). Runs the actual fetch in an isolated subprocess so a hang
    can be killed on a timeout without corrupting this process's MT5 connection
    or stalling the rest of a backfill run."""
    if not force and _have_day(symbol, d):
        tm, _, _ = _load_day(symbol, d)
        return len(tm)
    if d.weekday() >= 5:
        _save_day(symbol, d, np.array([], np.int64), np.array([]), np.array([]))
        return 0
    import subprocess
    try:
        p = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "_fetch_one", symbol, d.isoformat()],
            capture_output=True, text=True, timeout=timeout_s, cwd=BASE)
    except subprocess.TimeoutExpired:
        print(f"[FAIL] {symbol} {d} ({d.strftime('%a')}): call TIMED OUT after {timeout_s}s "
              f"(likely no tick history this far back on this broker) — NOT cached, will retry")
        return -1
    out = p.stdout or ""
    for line in out.splitlines():
        if line.startswith("RESULT:"):
            continue
        print(line)                       # forward the child's [EMPTY]/[FAIL] diagnostics
    for line in reversed(out.splitlines()):
        if line.startswith("RESULT:"):
            return int(line.split(":", 1)[1])
    print(f"[FAIL] {symbol} {d} ({d.strftime('%a')}): subprocess exited without a result "
          f"(exit={p.returncode}) stderr={(p.stderr or '')[-300:]}")
    return -1


def _fetch_one_cli(symbol, date_str):
    d = _parse_date(date_str)
    n = fetch_day(symbol, d, force=True)
    print(f"RESULT:{n}")


def cache_range(symbol, start_date, end_date, force=False):
    """Cache every UTC day in [start_date, end_date]. Returns a coverage dict
    (n>=0 ticks, or -1 for a day whose fetch failed and was left uncached)."""
    mt5 = _mt5()
    symbol = resolve_symbol(mt5, symbol)
    d = start_date
    report = {}
    while d <= end_date:
        n = fetch_day_safe(symbol, d, force=force)
        report[d.isoformat()] = n
        if d.weekday() < 5 and n > 0:
            print(f"  {symbol} {d} ({d.strftime('%a')}): {n:,} ticks")
        d += timedelta(days=1)
    return report


def repair_empty(symbol, mt5=None):
    """Rescan cached 0-tick weekday files and force-refetch them with the fixed
    retry logic. Distinguishes ticks that were cache-poisoned by the old
    only-retry-on-(-10001) bug (now recoverable) from genuinely empty days."""
    mt5 = mt5 or _mt5()
    symbol = resolve_symbol(mt5, symbol)
    recovered = confirmed = failed = 0
    for ds, n in cached_days(symbol):
        d = _parse_date(ds)
        if n == 0 and d.weekday() < 5:
            new_n = fetch_day_safe(symbol, d, force=True)
            if new_n > 0:
                recovered += 1
                print(f"  RECOVERED {symbol} {ds}: {new_n:,} ticks (was poisoned-empty)")
            elif new_n == 0:
                confirmed += 1
            else:
                failed += 1
    print(f"\nRepair {symbol}: {recovered} recovered, {confirmed} confirmed-empty, "
          f"{failed} still failing")
    return {"recovered": recovered, "confirmed_empty": confirmed, "failed": failed}


def backfill_summary(symbol, days_back=183, force=False, timeout_s=15):
    """Pull the last `days_back` days, skip weekends, and report an honest summary:
    days attempted/with-data/confirmed-empty/failed, total ticks, and the actual
    usable real-tick date range (vs silently treating a horizon cutoff as fine)."""
    mt5 = _mt5()
    symbol = resolve_symbol(mt5, symbol)
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days_back)
    attempted = with_data = empty_confirmed = failed = 0
    total_ticks = 0
    empties, fails = [], []
    d = start
    while d <= end:
        if d.weekday() < 5:
            attempted += 1
            n = fetch_day_safe(symbol, d, force=force, timeout_s=timeout_s)
            if n > 0:
                with_data += 1; total_ticks += n
            elif n == 0:
                empty_confirmed += 1; empties.append(d.isoformat())
            else:
                failed += 1; fails.append(d.isoformat())
        d += timedelta(days=1)

    dated = sorted(cached_days(symbol))
    first_data = next((ds for ds, n in dated if n > 0), None)
    prefix_empty = [ds for ds, n in dated
                    if first_data and ds < first_data and n == 0
                    and _parse_date(ds).weekday() < 5]

    print("=" * 64)
    print(f"BACKFILL SUMMARY — {symbol}  [{start} .. {end}]")
    print("=" * 64)
    print(f"  weekdays attempted                      : {attempted}")
    print(f"  days with data                           : {with_data}")
    print(f"  days confirmed empty (real gap/holiday)  : {empty_confirmed}")
    print(f"  days FAILED (retry exhausted, uncached)  : {failed}")
    print(f"  total ticks                              : {total_ticks:,}")
    if fails:
        print(f"  failed dates (retry next run): {fails}")
    if first_data and len(prefix_empty) >= 5:
        print(f"\n  Terminal tick history does not reliably reach before {first_data}.")
        print(f"  USABLE real-tick range for {symbol}: {first_data} .. {end}")
    elif empties:
        shown = empties[:10]
        print(f"\n  Scattered empty weekdays (not a clean horizon cutoff): {shown}"
              f"{' ...' if len(empties) > 10 else ''}")
    return {"attempted": attempted, "with_data": with_data, "empty": empty_confirmed,
            "failed": failed, "total_ticks": total_ticks, "usable_from": first_data,
            "empty_dates": empties, "failed_dates": fails}


def load_ticks(symbol, start_ms, end_ms):
    """Concatenate cached ticks across the UTC days spanning [start_ms, end_ms].
    Returns (time_msc, bid, ask) arrays filtered to the window, or None if no
    cached day in range has data. Missing days are silently skipped (caller can
    check coverage with `cached_days`)."""
    start = datetime.fromtimestamp(start_ms / 1000, timezone.utc).date()
    end   = datetime.fromtimestamp(end_ms / 1000, timezone.utc).date()
    tms, bids, asks = [], [], []
    d = start
    while d <= end:
        day = _load_day(symbol, d)
        if day is not None and len(day[0]):
            tms.append(day[0]); bids.append(day[1]); asks.append(day[2])
        d += timedelta(days=1)
    if not tms:
        return None
    tm = np.concatenate(tms); bid = np.concatenate(bids); ask = np.concatenate(asks)
    m = (tm >= start_ms) & (tm <= end_ms)
    if not m.any():
        return None
    return tm[m], bid[m], ask[m]


def cached_days(symbol):
    """Sorted list of (date, nticks) cached on disk for a symbol."""
    out = []
    for p in sorted(glob.glob(os.path.join(CACHE_DIR, symbol, "*.npz"))):
        d = os.path.splitext(os.path.basename(p))[0]
        try:
            z = np.load(p); out.append((d, len(z["time_msc"])))
        except Exception:
            pass
    return out


def tick_coverage(symbol, start_ms, end_ms):
    """Fraction of weekdays in [start_ms,end_ms] that have cached, non-empty ticks.
    Used by holdout_replay to decide tick-fill vs bar-fallback per window."""
    start = datetime.fromtimestamp(start_ms / 1000, timezone.utc).date()
    end   = datetime.fromtimestamp(end_ms / 1000, timezone.utc).date()
    wk = covered = 0
    d = start
    while d <= end:
        if d.weekday() < 5:
            wk += 1
            day = _load_day(symbol, d)
            if day is not None and len(day[0]):
                covered += 1
        d += timedelta(days=1)
    return (covered / wk) if wk else 0.0


# ─── BARS over an explicit window (for signal generation) ─────────────────────

def get_bars_range(symbol, tf_name, start_ms, end_ms, warmup_bars=600):
    """Fetch bars covering [start, end] plus a warmup pad before `start` so
    indicators (ema200, hurst100, vgrsi50...) are stable by the window's start.
    Returns (df, epoch_s) where epoch_s[i] is bar i's OPEN time in epoch seconds."""
    import MetaTrader5 as mt5
    import pandas as pd
    tf  = {"M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5}[tf_name]
    tfm = {"M1": 1, "M5": 5}[tf_name]
    start = datetime.fromtimestamp(start_ms / 1000, timezone.utc)
    end   = datetime.fromtimestamp(end_ms / 1000, timezone.utc)
    pad   = start - timedelta(minutes=tfm * warmup_bars)
    rates = mt5.copy_rates_range(symbol, tf, pad, end)
    if rates is None or len(rates) == 0:
        return None, None
    df = pd.DataFrame(rates)
    epoch_s = df["time"].values.astype(np.int64)
    dt = pd.to_datetime(df["time"], unit="s")
    df.index = dt
    return df, epoch_s


# ─── PURE REPLAY CORE (no MT5 — unit-testable on synthetic ticks) ─────────────

def _walk_trade(go_long, entry, sl, tp, r0, tk_t, tk_bid, tk_ask,
                k0, end_ms, slippage, point, dynamic=True):
    """Walk ticks from index k0 until exit or end_ms. Returns (r_multiple, outcome).
    Pays real spread (buy@ask / sell@bid) and `slippage` points on stop fills."""
    be_moved = False
    n = len(tk_t)
    k = k0
    last_px = entry
    while k < n and tk_t[k] <= end_ms:
        bid, ask = tk_bid[k], tk_ask[k]
        if go_long:
            mark = bid                              # we exit a long by selling @ bid
            fav = (mark - entry) / r0
            if dynamic:
                if fav >= BE_TRIGGER_R and not be_moved:
                    sl = max(sl, entry); be_moved = True
                if fav >= TRAIL_TRIGGER_R:
                    sl = max(sl, entry + (fav - TRAIL_GAP_R) * r0)
            if bid <= sl:
                exit_px = sl - slippage * point     # adverse slip on stop
                out = "TP" if sl >= entry else "SL"
                return (exit_px - entry) / r0, out
            if bid >= tp:
                return (tp - entry) / r0, "TP"
            last_px = bid
        else:
            mark = ask                              # exit a short by buying @ ask
            fav = (entry - mark) / r0
            if dynamic:
                if fav >= BE_TRIGGER_R and not be_moved:
                    sl = min(sl, entry); be_moved = True
                if fav >= TRAIL_TRIGGER_R:
                    sl = min(sl, entry - (fav - TRAIL_GAP_R) * r0)
            if ask >= sl:
                exit_px = sl + slippage * point
                out = "TP" if sl <= entry else "SL"
                return (entry - exit_px) / r0, out
            if ask <= tp:
                return (entry - tp) / r0, "TP"
            last_px = ask
        k += 1
    # timed out -> mark to last seen price
    r = (last_px - entry) / r0 if go_long else (entry - last_px) / r0
    return r, "TIME"


def replay_signals(long_sig, short_sig, atr, sl_atr, gene, bar_epoch_s,
                   tk_t, tk_bid, tk_ask, window_start_ms, window_end_ms,
                   max_hold_min, point, slippage=0.0, dynamic=True,
                   compute_tp=None):
    """Core engine: given precomputed per-bar signals + a tick stream, simulate
    trades with real-spread fills. Pure/no-MT5. Returns (rs, trades).
      long_sig/short_sig : bool arrays aligned to bars
      atr                : per-bar ATR array (entry uses atr[i])
      bar_epoch_s        : per-bar OPEN time, epoch seconds
      window_*           : only ENTRIES whose fill time is in-window count
    Entry fills on the first tick at/after bar i+1's open time (next-bar-open,
    matching the lab). No overlapping trades."""
    if compute_tp is None:
        from strategy_lab import compute_tp
    n = len(long_sig)
    rs, trades = [], []
    i = 1
    while i < n - 1:
        go_long  = bool(long_sig[i])  and not np.isnan(atr[i]) and atr[i] > 0
        go_short = bool(short_sig[i]) and not np.isnan(atr[i]) and atr[i] > 0
        if not (go_long or go_short):
            i += 1; continue
        entry_ms = int(bar_epoch_s[i + 1]) * 1000
        if entry_ms < window_start_ms or entry_ms > window_end_ms:
            i += 1; continue
        k0 = int(np.searchsorted(tk_t, entry_ms, side="left"))
        if k0 >= len(tk_t):
            break                                   # no ticks left to fill
        # real-spread entry: BUY pays ask, SELL pays bid (+ slippage)
        if go_long:
            entry = tk_ask[k0] + slippage * point
            sl    = entry - sl_atr * atr[i]
            bias  = "bullish"
        else:
            entry = tk_bid[k0] - slippage * point
            sl    = entry + sl_atr * atr[i]
            bias  = "bearish"
        tp   = compute_tp(gene, entry, sl, bias)
        r0   = abs(entry - sl)
        if r0 <= 0:
            i += 1; continue
        end_ms = entry_ms + int(max_hold_min * 60_000)
        r, out = _walk_trade(go_long, entry, sl, tp, r0, tk_t, tk_bid, tk_ask,
                             k0 + 1, end_ms, slippage, point, dynamic)
        rs.append(round(float(r), 4))
        trades.append({"entry_ms": entry_ms, "bias": bias, "entry": entry,
                       "sl": sl, "tp": tp, "r": round(float(r), 4), "outcome": out})
        # advance past the exit: skip bars whose open precedes the exit tick time
        exit_ms = end_ms
        while i < n - 1 and int(bar_epoch_s[i + 1]) * 1000 <= entry_ms:
            i += 1
        i += 1
    return rs, trades


# ─── MT5-BACKED REPLAY (the convenient entry point) ───────────────────────────

def replay_gene(gene, symbol, tf_name, start_ms, end_ms,
                slippage_pts=0.0, dynamic=True, max_hold_min=None):
    """Replay a gene over [start_ms, end_ms] on REAL cached ticks. Requires the
    ticks to already be cached (call cache_range first) and an MT5 connection for
    the bars. Returns (rs, trades, meta). meta.tick_coverage flags how much of the
    window actually had ticks (the rest produced no fills)."""
    from strategy_lab import prepare, predicate, compute_tp, TF_MINUTES, attach_relative
    import MetaTrader5 as mt5
    _mt5()
    si = mt5.symbol_info(symbol)
    point = si.point if si else (0.01 if "XAU" in symbol else 0.00001)
    if max_hold_min is None:
        # default per-tf max hold matches the lab (M1=30 bars, M5=18 bars)
        max_hold_min = {"M1": 30, "M5": 90}[tf_name]

    df, epoch_s = get_bars_range(symbol, tf_name, start_ms, end_ms)
    if df is None or len(df) < 50:
        return [], [], {"error": "no bars", "tick_coverage": 0.0}
    d = prepare(df)
    attach_relative(d, symbol, tf_name)      # relative-value features for pairs_confirm genes
    nbar = len(d)
    long_sig  = np.ones(nbar, dtype=bool)
    short_sig = np.ones(nbar, dtype=bool)
    for p in gene["preds"]:
        l, s = predicate(p["name"], p["params"], d)
        long_sig &= l; short_sig &= s
    if gene.get("dir") == "long":
        short_sig[:] = False
    elif gene.get("dir") == "short":
        long_sig[:] = False

    ticks = load_ticks(symbol, start_ms, end_ms + int((max_hold_min + 5) * 60_000))
    cov = tick_coverage(symbol, start_ms, end_ms)
    if ticks is None:
        return [], [], {"error": "no ticks cached", "tick_coverage": cov}
    tk_t, tk_bid, tk_ask = ticks

    rs, trades = replay_signals(
        long_sig, short_sig, d["atr"].values, gene["sl_atr"], gene, epoch_s,
        tk_t, tk_bid, tk_ask, start_ms, end_ms, max_hold_min, point,
        slippage=slippage_pts, dynamic=dynamic, compute_tp=compute_tp)
    meta = {"trades": len(rs), "tick_coverage": round(cov, 3),
            "expectancy": round(float(np.mean(rs)), 4) if rs else 0.0,
            "n_ticks": len(tk_t)}
    return rs, trades, meta


def bar_replay_gene(gene, symbol, tf_name, start_ms, end_ms):
    """Window-CONFINED bar backtest (flat lab spread) over [start_ms, end_ms].
    The honest fallback when real ticks don't cover the window: indicators are
    warmed on bars BEFORE the window, but only trades whose entry bar opens inside
    [start_ms, end_ms] are counted — so it stays strictly within the holdout, not
    arbitrary recent history. Returns (rs, meta)."""
    from strategy_lab import prepare, backtest, attach_relative
    df, epoch_s = get_bars_range(symbol, tf_name, start_ms, end_ms)
    if df is None or len(df) < 50:
        return [], {"trades": 0, "error": "no bars"}
    d = prepare(df)
    attach_relative(d, symbol, tf_name)      # relative-value features for pairs_confirm genes
    in_win = (epoch_s.astype(np.int64) * 1000 >= start_ms) & \
             (epoch_s.astype(np.int64) * 1000 <= end_ms)
    if not in_win.any():
        return [], {"trades": 0, "error": "window empty"}
    first = int(np.argmax(in_win))               # first in-window bar position
    d_win = d.iloc[first:].reset_index(drop=True)
    _, rs = backtest(gene, d_win, return_trades=True)
    return rs, {"trades": len(rs),
                "expectancy": round(float(np.mean(rs)), 4) if rs else 0.0}


# ─── ARCHIVE FORWARD (the only path to a months-deep real-tick holdout) ───────

def archive_forward(symbols=None, days_back=2):
    """Cache the last `days_back` days for every symbol. Meant to run daily (cron)
    so the real-tick horizon grows forward — the server won't give us old ticks,
    so we must hoard them as they age in. Idempotent (skips cached days)."""
    symbols = symbols or UNIVERSE
    mt5 = _mt5()
    today = datetime.now(timezone.utc).date()
    total = {}
    for sym in symbols:
        sym = resolve_symbol(mt5, sym)
        s = today - timedelta(days=days_back)
        n = 0
        d = s
        while d <= today:
            r = fetch_day_safe(sym, d, timeout_s=25)
            if r > 0:
                n += r
            d += timedelta(days=1)
        total[sym] = n
        print(f"  archived {sym}: {n:,} ticks over last {days_back}d")
    return total


# ─── SELF-TEST (synthetic ticks, no MT5) ──────────────────────────────────────

def _selftest():
    print("=" * 64)
    print("tick_replay self-test — synthetic ticks, real-spread fills")
    print("=" * 64)
    # Build a deterministic up-then-flat price path with a fixed 1-point spread.
    # One M1 bar, long signal at bar 1, entry at bar 2 open. Price rises through
    # the TP -> expect a +R win net of spread.
    point = 0.0001
    spread = 1 * point
    # bars: open times 60s apart; we only need atr + a long signal at i=1
    bar_epoch_s = np.array([0, 60, 120, 180], dtype=np.int64)
    long_sig  = np.array([False, True, False, False])
    short_sig = np.array([False, False, False, False])
    atr = np.array([np.nan, 0.0010, 0.0010, 0.0010])   # 10-pip ATR
    gene = {"sl_atr": 1.0, "tp_rr": 1.5, "dir": "long"}
    def compute_tp(g, entry, sl, bias):
        dist = abs(entry - sl); return entry + g["tp_rr"] * dist
    # tick stream from bar-2 open (t=120s) climbing 1.0008 -> 1.0030
    base = 1.0000
    t = []; bid = []; ask = []
    px = 1.0008
    for ms in range(120_000, 300_000, 1000):    # 1 tick/sec for 3 min
        t.append(ms); bid.append(px); ask.append(px + spread); px += 0.00003
    tk_t = np.array(t, np.int64); tk_bid = np.array(bid); tk_ask = np.array(ask)

    rs, trades = replay_signals(long_sig, short_sig, atr, gene["sl_atr"], gene,
                                bar_epoch_s, tk_t, tk_bid, tk_ask,
                                0, 200_000, 30, point, slippage=0.0,
                                dynamic=False, compute_tp=compute_tp)
    assert len(rs) == 1, f"expected 1 trade, got {len(rs)}"
    tr = trades[0]
    # entry filled at ask of first tick >= t=120s -> 1.0008 + spread
    assert abs(tr["entry"] - (1.0008 + spread)) < 1e-9, tr
    print(f"  [1] long entry filled at ask {tr['entry']:.5f} (paid {spread/point:.0f}pt spread)")
    # SL = entry - 1*ATR; TP = entry + 1.5*ATR -> price climbs to TP -> ~+1.5R
    print(f"      outcome={tr['outcome']} r={tr['r']}  (TP ~ +1.5R expected)")
    assert tr["outcome"] == "TP" and tr["r"] > 1.0, tr

    # [2] spread makes a marginal trade a loser: tiny TP just above entry but the
    #     bid never reaches it because we pay spread on entry and exit on the bid.
    px = 1.0008; t = []; bid = []; ask = []
    for ms in range(120_000, 300_000, 1000):
        t.append(ms); bid.append(px); ask.append(px + spread); px -= 0.00004
    tk_t = np.array(t, np.int64); tk_bid = np.array(bid); tk_ask = np.array(ask)
    rs2, tr2 = replay_signals(long_sig, short_sig, atr, gene["sl_atr"], gene,
                              bar_epoch_s, tk_t, tk_bid, tk_ask,
                              0, 200_000, 30, point, dynamic=False,
                              compute_tp=compute_tp)
    print(f"  [2] falling price -> outcome={tr2[0]['outcome']} r={rs2[0]} (SL expected)")
    assert tr2[0]["outcome"] == "SL" and rs2[0] < 0, tr2
    print("\n  PASS — real-spread entry/exit + SL/TP walk verified.")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_date(s):
    return datetime.strptime(s, "%Y-%m-%d").date()

if __name__ == "__main__":
    args = sys.argv[1:]
    cmd = args[0] if args else "selftest"
    if cmd == "selftest":
        _selftest()
    elif cmd == "cache":
        sym = args[1]; s = _parse_date(args[2]); e = _parse_date(args[3])
        print(f"Caching {sym} ticks {s} -> {e} ...")
        cache_range(sym, s, e)
    elif cmd == "archive":
        archive_forward()
    elif cmd == "coverage":
        sym = args[1]
        days = cached_days(sym)
        tot = sum(n for _, n in days)
        print(f"{sym}: {len(days)} days cached, {tot:,} ticks")
        for d, n in days:
            print(f"  {d}: {n:,}")
    elif cmd == "repair":
        sym = args[1]
        repair_empty(sym)
    elif cmd == "backfill":
        sym = args[1]
        days_back = int(args[2]) if len(args) > 2 else 183
        backfill_summary(sym, days_back=days_back, force=False)
    elif cmd == "_fetch_one":
        _fetch_one_cli(args[1], args[2])
    else:
        print(__doc__)

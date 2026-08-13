"""
CPCV — Combinatorial Purged Cross-Validation for the strategy lab
=================================================================
Phase 2B, priority 4. The lab validates a gene on ONE 70/30 split: a single
out-of-sample path, one number, easy to get lucky on. Lopez de Prado's CPCV
splits the bars into N groups, holds out k of them as test in EVERY combination
(C(N,k) of them), and evaluates each — yielding many OOS paths and a DISTRIBUTION
of out-of-sample Sharpe instead of a point estimate. A gene that only looks good
on one slicing is exposed.

THE LEAK THIS PREVENTS (load-bearing, per the brief): a scalp trade entered at
bar i is not resolved until up to `max_hold` bars later, so its label spans
[i, i+max_hold]. If a test segment's trade reaches forward into bars that belong
to an adjacent (train) group, the test "borrows" future information and every
metric inflates. So:
  * PURGE  — a test entry is only counted if the WHOLE trade [i, i+max_hold]
             lies inside the contiguous test segment (no reaching into a
             neighbouring group). Straddling entries are dropped.
  * EMBARGO — additionally drop the first `embargo` bars after each train->test
             transition, killing serial-correlation bleed across the seam.

This is evaluation of a FIXED rule-gene across folds (the gene carries no fitted
parameters), so there is no train-side refit; the purge/embargo are applied to
keep each OOS path honestly contained. The sim loop is a faithful copy of
strategy_lab.backtest (same next-bar-open fill, same SL/TP/timeout, same
SPREAD_COST_R) so CPCV numbers are directly comparable to the lab's.

Self-test:  py cpcv.py
"""

from __future__ import annotations
import sys, math
from itertools import combinations
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Defaults: 6 groups, 2 held out per combination -> C(6,2)=15 OOS paths.
N_GROUPS_DEFAULT = 6
K_TEST_DEFAULT   = 2
EMBARGO_FRAC     = 0.01      # 1% of bars embargoed after each test transition

try:
    from strategy_lab import SPREAD_COST_R
except Exception:
    SPREAD_COST_R = 0.08


# ─── INDEXED SIM (faithful copy of strategy_lab.backtest, records bar indices) ─

def indexed_trades(gene, d, compute_tp=None, predicate=None):
    """Run the gene over a prepared frame and return a list of
    (entry_i, exit_i, r) — same simulation as strategy_lab.backtest but with the
    entry/exit BAR INDICES so trades can be assigned to CV folds and purged."""
    if compute_tp is None:
        from strategy_lab import compute_tp
    if predicate is None:
        from strategy_lab import predicate
    n = len(d)
    long_sig  = np.ones(n, dtype=bool)
    short_sig = np.ones(n, dtype=bool)
    for p in gene["preds"]:
        l, s = predicate(p["name"], p["params"], d)
        long_sig &= l; short_sig &= s
    if gene.get("dir") == "long":
        short_sig[:] = False
    elif gene.get("dir") == "short":
        long_sig[:] = False

    o   = d["open"].values
    hi  = d["high"].values
    lo  = d["low"].values
    cl  = d["close"].values
    atr = d["atr"].values
    max_hold = gene.get("max_hold", 18)

    out = []
    i = 1
    while i < n - 1:
        go_long  = long_sig[i]  and not np.isnan(atr[i]) and atr[i] > 0
        go_short = short_sig[i] and not np.isnan(atr[i]) and atr[i] > 0
        if not (go_long or go_short):
            i += 1; continue
        entry = o[i + 1]
        if go_long:
            sl = entry - gene["sl_atr"] * atr[i]; tp = compute_tp(gene, entry, sl, "bullish")
        else:
            sl = entry + gene["sl_atr"] * atr[i]; tp = compute_tp(gene, entry, sl, "bearish")
        risk = abs(entry - sl)
        if risk <= 0:
            i += 1; continue
        tp_r = abs(tp - entry) / risk
        outcome = None
        j = i + 1
        end = min(n, j + max_hold)
        while j < end:
            if go_long:
                if lo[j] <= sl: outcome = -1.0; break
                if hi[j] >= tp: outcome = tp_r; break
            else:
                if hi[j] >= sl: outcome = -1.0; break
                if lo[j] <= tp: outcome = tp_r; break
            j += 1
        if outcome is None:
            last = cl[min(j, n - 1)]
            outcome = ((last - entry) if go_long else (entry - last)) / risk
        out.append((i + 1, min(j, n - 1), outcome - SPREAD_COST_R))   # entry bar = i+1
        i = j + 1
    return out


# ─── GROUPS / COMBINATORIAL SPLITS ────────────────────────────────────────────

def make_groups(n_bars, n_groups):
    """Contiguous, near-equal index blocks: [(start,end_exclusive), ...]."""
    edges = np.linspace(0, n_bars, n_groups + 1, dtype=int)
    return [(int(edges[g]), int(edges[g + 1])) for g in range(n_groups)]


def _contiguous_segments(test_group_ids, groups):
    """Merge chosen test groups into contiguous [start,end) segments."""
    ids = sorted(test_group_ids)
    segs = []
    cur_s, cur_e = groups[ids[0]]
    for gid in ids[1:]:
        s, e = groups[gid]
        if s == cur_e:                # adjacent -> extend the segment
            cur_e = e
        else:
            segs.append((cur_s, cur_e)); cur_s, cur_e = s, e
    segs.append((cur_s, cur_e))
    return segs


# ─── CPCV EVALUATION ──────────────────────────────────────────────────────────

def cpcv_evaluate(gene, d, n_groups=N_GROUPS_DEFAULT, k=K_TEST_DEFAULT,
                  embargo_frac=EMBARGO_FRAC, max_hold=None, purge=True):
    """Evaluate `gene` across all C(n_groups, k) combinatorial OOS paths with
    purge+embargo. Returns a dict with the per-path metrics and the OOS Sharpe /
    expectancy DISTRIBUTION. Set purge=False to see the (inflated) naive numbers."""
    n = len(d)
    if max_hold is None:
        max_hold = gene.get("max_hold", 18)
    embargo = int(n * embargo_frac)
    groups = make_groups(n, n_groups)
    trades = indexed_trades(gene, d)              # (entry_i, exit_i, r) once, reuse

    paths = []
    for test_ids in combinations(range(n_groups), k):
        segs = _contiguous_segments(test_ids, groups)
        rs = []
        for (a, b) in segs:
            lo_entry = a + (embargo if purge else 0)     # embargo after the seam
            for (ei, xi, r) in trades:
                if ei < lo_entry or ei >= b:
                    continue                              # entry not in test segment
                if purge and (ei + max_hold) >= b:
                    continue                              # PURGE: trade reaches past seg
                rs.append(r)
        if len(rs) >= 2:
            arr = np.array(rs)
            sd = arr.std(ddof=1)
            paths.append({
                "test_groups": list(test_ids),
                "trades": len(rs),
                "expectancy": float(arr.mean()),
                "sharpe": float(arr.mean() / sd) if sd > 0 else 0.0,
                "total_r": float(arr.sum()),
            })

    if not paths:
        return {"n_paths": 0, "trades_total": 0, "purge": purge}
    sh = np.array([p["sharpe"] for p in paths])
    ex = np.array([p["expectancy"] for p in paths])
    return {
        "n_paths": len(paths),
        "trades_per_path_med": int(np.median([p["trades"] for p in paths])),
        "sharpe_mean": round(float(sh.mean()), 4),
        "sharpe_std":  round(float(sh.std(ddof=1)), 4) if len(sh) > 1 else 0.0,
        "sharpe_q05":  round(float(np.quantile(sh, 0.05)), 4),
        "sharpe_q50":  round(float(np.quantile(sh, 0.50)), 4),
        "exp_mean":    round(float(ex.mean()), 4),
        "exp_q05":     round(float(np.quantile(ex, 0.05)), 4),
        "frac_paths_positive": round(float((ex > 0).mean()), 3),
        "purge": purge, "embargo": embargo, "max_hold": max_hold,
        "paths": paths,
    }


def cpcv_validate(gene, d, min_frac_positive=0.6, min_sharpe_q05=0.0, **kw):
    """A robustness gate to replace the lab's single 70/30 OOS check: a gene
    passes only if it is positive on a strong MAJORITY of OOS paths and its 5th-
    percentile path Sharpe clears a floor (i.e. even unlucky slicings hold up).
    Returns (passed, summary)."""
    s = cpcv_evaluate(gene, d, **kw)
    if s.get("n_paths", 0) == 0:
        return False, s
    passed = (s["frac_paths_positive"] >= min_frac_positive and
              s["sharpe_q05"] >= min_sharpe_q05)
    s["passed"] = bool(passed)
    return passed, s


# ─── SELF-TEST ────────────────────────────────────────────────────────────────

def _synthetic_frame(n=3000, seed=0):
    """Build a prepared-like frame the indexed sim can run on. A mild upward
    drift with autocorrelation so trades that straddle a fold boundary borrow
    real future info — exactly what purge must remove."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 1.0, n)
    steps = np.convolve(steps, np.ones(5) / 5, mode="same")   # autocorrelation
    price = 100 + np.cumsum(steps) * 0.02
    high = price + np.abs(rng.normal(0, 0.05, n))
    low  = price - np.abs(rng.normal(0, 0.05, n))
    d = pd.DataFrame({"open": price, "high": high, "low": low, "close": price})
    d["atr"] = (d["high"] - d["low"]).rolling(14).mean().bfill()
    return d


def _selftest():
    print("=" * 68)
    print("CPCV self-test")
    print("=" * 68)
    d = _synthetic_frame()
    # a simple always-long breakout-ish gene via a momentum predicate stand-in:
    # use ema_trend so it actually fires on the synthetic frame.
    import pandas as pd
    for span in (9, 20, 50, 100, 200):
        d[f"ema{span}"] = d["close"].ewm(span=span, adjust=False).mean()
    gene = {"preds": [{"name": "ema_trend", "params": {"fast": 9, "slow": 50}}],
            "sl_atr": 1.0, "tp_rr": 1.5, "dir": "long", "max_hold": 20}

    naive  = cpcv_evaluate(gene, d, purge=False)
    purged = cpcv_evaluate(gene, d, purge=True)
    print(f"\n  paths: {purged['n_paths']} (C(6,2)), "
          f"~{purged['trades_per_path_med']} trades/path")
    print(f"  NAIVE  (no purge): Sharpe mean {naive['sharpe_mean']:+.3f} "
          f"q05 {naive['sharpe_q05']:+.3f}  exp_mean {naive['exp_mean']:+.4f}  "
          f"frac+ {naive['frac_paths_positive']}")
    print(f"  PURGED+EMBARGO   : Sharpe mean {purged['sharpe_mean']:+.3f} "
          f"q05 {purged['sharpe_q05']:+.3f}  exp_mean {purged['exp_mean']:+.4f}  "
          f"frac+ {purged['frac_paths_positive']}")
    assert purged["n_paths"] == 15, "C(6,2) must be 15 paths"
    # purge should never INCREASE the trade count vs naive (it only drops straddlers)
    assert purged["trades_per_path_med"] <= naive["trades_per_path_med"], \
        "purge must not add trades"
    print("\n  Distribution of OOS Sharpe across the 15 paths (purged):")
    sh = sorted(p["sharpe"] for p in purged["paths"])
    print("    " + "  ".join(f"{x:+.2f}" for x in sh))
    print("\n  PASS — purge drops boundary-straddling trades; many OOS paths give "
          "a Sharpe distribution, not one lucky number.")


if __name__ == "__main__":
    _selftest()

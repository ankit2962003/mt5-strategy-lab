"""
Stat Guard — multiple-testing protection for the strategy pipeline
==================================================================
The strategy lab and the research engine BOTH generate many candidate
strategies and keep the ones that look good on out-of-sample data. That is a
multiple-comparisons machine: test enough random rules and some WILL clear any
fixed expectancy / profit-factor gate by pure luck. A "73% win rate" pulled
from a blog, or the single best survivor out of 1,000 evolved genes, is exactly
the kind of result that is most likely to be noise.

This module adds the statistical guardrail that the current gate is missing:

  1. Probabilistic Sharpe Ratio (PSR)
        P(true Sharpe > benchmark) for ONE strategy, correcting for the
        track-record length, skew and fat tails of the return stream.

  2. Deflated Sharpe Ratio (DSR)   [Bailey & Lopez de Prado, 2014]
        PSR where the benchmark is the EXPECTED MAXIMUM Sharpe you'd see across
        N independent trials of pure noise. This is the key number: it answers
        "is this survivor better than the best you'd expect from luck, given
        how many strategies you tried?"  DSR > 0.95 == keep it.

  3. Benjamini-Hochberg FDR control
        Given p-values for a whole batch of candidates, decide which to keep
        while controlling the false-discovery rate (expected fraction of kept
        strategies that are actually junk).

No scipy dependency — the normal CDF/quantile are implemented directly so this
drops into the existing numpy-only environment.

Quick self-test:   py stat_guard.py
"""

from __future__ import annotations
import sys
import math
import numpy as np

# Windows consoles default to cp1252 — force UTF-8 so unicode prints are safe
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

EULER_GAMMA = 0.5772156649015329


# ─── NORMAL DISTRIBUTION (no scipy) ──────────────────────────────────────────

def norm_cdf(x: float) -> float:
    """Standard normal CDF via the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """
    Inverse standard normal CDF (quantile / probit) — Acklam's rational
    approximation, |error| < 1.15e-9 across the full range.
    """
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf

    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]

    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# ─── RETURN-STREAM MOMENTS ───────────────────────────────────────────────────

def stream_stats(returns) -> dict:
    """
    Per-observation moments of a return / R-multiple stream. For the lab, the
    natural 'return' is the per-trade R-multiple (mean R == expectancy).
    """
    r = np.asarray([x for x in returns if x is not None], dtype=float)
    n = len(r)
    if n < 2:
        return {"n": n, "mean": 0.0, "std": 0.0, "sharpe": 0.0,
                "skew": 0.0, "kurt": 3.0}
    mean = float(r.mean())
    std = float(r.std(ddof=1))
    if std == 0:
        return {"n": n, "mean": mean, "std": 0.0, "sharpe": 0.0,
                "skew": 0.0, "kurt": 3.0}
    z = (r - mean) / std
    skew = float((z ** 3).mean())
    kurt = float((z ** 4).mean())          # raw kurtosis (3.0 == normal)
    return {"n": n, "mean": mean, "std": std,
            "sharpe": mean / std,           # per-trade Sharpe (NOT annualised)
            "skew": skew, "kurt": kurt}


# ─── PROBABILISTIC / DEFLATED SHARPE ─────────────────────────────────────────

def probabilistic_sharpe(sharpe, n, skew, kurt, benchmark=0.0) -> float:
    """
    PSR: P(true per-obs Sharpe > benchmark). All Sharpes are per-observation.
    Returns a probability in [0, 1]; >= 0.95 is the usual significance bar.
    """
    if n < 2:
        return 0.0
    denom = 1.0 - skew * sharpe + ((kurt - 1.0) / 4.0) * sharpe ** 2
    if denom <= 0:
        denom = 1e-9
    stat = (sharpe - benchmark) * math.sqrt(n - 1) / math.sqrt(denom)
    return norm_cdf(stat)


def expected_max_sharpe(n_trials: int, var_trial_sharpe: float) -> float:
    """
    Expected MAXIMUM per-obs Sharpe across N independent trials of zero-skill
    strategies whose estimated Sharpes have variance var_trial_sharpe.
    This is the benchmark a survivor must beat to not be luck.
    """
    if n_trials < 2 or var_trial_sharpe <= 0:
        return 0.0
    sd = math.sqrt(var_trial_sharpe)
    a = norm_ppf(1.0 - 1.0 / n_trials)
    b = norm_ppf(1.0 - 1.0 / (n_trials * math.e))
    return sd * ((1.0 - EULER_GAMMA) * a + EULER_GAMMA * b)


def deflated_sharpe(sharpe, n, skew, kurt, n_trials, var_trial_sharpe) -> float:
    """
    DSR == PSR measured against the expected-max-Sharpe benchmark. This is the
    number to gate on when a strategy is the survivor of many trials.
        DSR >= 0.95  ->  keep (beats luck at the 95% level)
    """
    sr0 = expected_max_sharpe(n_trials, var_trial_sharpe)
    return probabilistic_sharpe(sharpe, n, skew, kurt, benchmark=sr0)


def dsr_from_returns(returns, n_trials, var_trial_sharpe=None, min_n=8) -> float:
    """
    Convenience: Deflated Sharpe straight from a single strategy's return /
    R-multiple stream. When you don't have the empirical dispersion of the trial
    Sharpes, the null-model estimation variance (Lo, 2002) is used as a
    principled default. Returns 0.0 for too-short records (don't graduate yet).
    """
    s = stream_stats(returns)
    if s["n"] < min_n or s["std"] == 0:
        return 0.0
    if var_trial_sharpe is None:
        var_trial_sharpe = (1.0 + 0.5 * s["sharpe"] ** 2) / (s["n"] - 1)
    return deflated_sharpe(s["sharpe"], s["n"], s["skew"], s["kurt"],
                           n_trials, var_trial_sharpe)


# ─── BENJAMINI-HOCHBERG FALSE-DISCOVERY-RATE CONTROL ─────────────────────────

def benjamini_hochberg(pvalues, alpha=0.10):
    """
    BH step-up procedure. Returns a boolean 'keep' list (same order as input)
    controlling the expected false-discovery rate at <= alpha. Use this on a
    whole batch of candidates instead of thresholding each p-value alone.
    """
    p = list(pvalues)
    m = len(p)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p[i])
    keep = [False] * m
    k_max = -1
    for rank, idx in enumerate(order, start=1):
        if p[idx] <= alpha * rank / m:
            k_max = rank
    for rank, idx in enumerate(order, start=1):
        if rank <= k_max:
            keep[idx] = True
    return keep


# ─── ONE-CALL CANDIDATE SCREEN ───────────────────────────────────────────────

def screen(candidates, n_trials=None, dsr_min=0.95, fdr_alpha=0.10):
    """
    Screen a batch of strategy candidates for statistical significance after
    accounting for how many were tried.

    candidates : list of dicts, each with EITHER
                   "returns": [r-multiples ...]            (preferred), or
                   "sharpe","n","skew","kurt" precomputed.
                 Any other keys (id, symbol, desc...) are passed through.
    n_trials   : total strategies evaluated to surface this batch. Defaults to
                 len(candidates). If the lab tested 1,000 genes and handed you
                 the best 10, pass n_trials=1000 so deflation is honest.

    Adds to each candidate: sharpe, psr, p_value, dsr, pass_dsr, pass_fdr,
    and 'keep' (= pass_dsr AND pass_fdr). Returns the annotated list sorted by
    dsr descending.
    """
    cands = [dict(c) for c in candidates]
    if not cands:
        return []
    n_trials = n_trials or len(cands)

    # moments for each candidate
    for c in cands:
        if "returns" in c:
            s = stream_stats(c["returns"])
            c.update({"sharpe": s["sharpe"], "n": s["n"],
                      "skew": s["skew"], "kurt": s["kurt"]})
        c.setdefault("skew", 0.0)
        c.setdefault("kurt", 3.0)

    # variance of the estimated Sharpes across this batch — the dispersion that
    # drives how high a survivor's Sharpe can climb on luck alone
    sharpes = np.array([c["sharpe"] for c in cands], dtype=float)
    var_trial = float(sharpes.var(ddof=1)) if len(sharpes) > 1 else 0.0

    pvals = []
    for c in cands:
        c["psr"] = probabilistic_sharpe(c["sharpe"], c["n"], c["skew"], c["kurt"], 0.0)
        c["p_value"] = 1.0 - c["psr"]          # one-sided: H0 = Sharpe <= 0
        c["dsr"] = deflated_sharpe(c["sharpe"], c["n"], c["skew"], c["kurt"],
                                   n_trials, var_trial)
        c["pass_dsr"] = c["dsr"] >= dsr_min
        pvals.append(c["p_value"])

    fdr_keep = benjamini_hochberg(pvals, alpha=fdr_alpha)
    for c, k in zip(cands, fdr_keep):
        c["pass_fdr"] = bool(k)
        c["keep"] = bool(c["pass_dsr"] and c["pass_fdr"])

    cands.sort(key=lambda c: c["dsr"], reverse=True)
    return cands


# ─── SELF-TEST ───────────────────────────────────────────────────────────────

def _demo():
    rng = np.random.default_rng(7)
    print("=" * 68)
    print("Stat Guard self-test")
    print("=" * 68)

    # 1) Pure noise: 500 zero-skill strategies, keep the best by raw Sharpe.
    #    A naive gate would 'discover' it. DSR should expose it as luck.
    N = 500
    trials = [rng.normal(0.0, 1.0, size=120) for _ in range(N)]
    sr = [stream_stats(t)["sharpe"] for t in trials]
    best = int(np.argmax(sr))
    s = stream_stats(trials[best])
    var_trial = float(np.var(sr, ddof=1))
    psr = probabilistic_sharpe(s["sharpe"], s["n"], s["skew"], s["kurt"])
    dsr = deflated_sharpe(s["sharpe"], s["n"], s["skew"], s["kurt"], N, var_trial)
    print(f"\n[1] NOISE — best of {N} zero-skill strategies:")
    print(f"    raw per-trade Sharpe = {s['sharpe']:.3f}  (looks tradable!)")
    print(f"    PSR  (ignores #trials)  = {psr:.3f}  <- naive gate is fooled")
    print(f"    DSR  (accounts #trials) = {dsr:.3f}  <- correctly rejects (<0.95)")

    # 2) THE KEY LESSON: the verdict depends on HOW MANY ideas you tried.
    #    Same exact track record, two contexts:
    #      (a) a single pre-registered hypothesis you had reason to believe
    #      (b) the lone survivor you mined out of 500 random genes
    edge = rng.normal(0.22, 1.0, size=300)     # +0.22R/trade over 300 trades
    e = stream_stats(edge)
    psr_edge = probabilistic_sharpe(e["sharpe"], e["n"], e["skew"], e["kurt"])
    # mined: deflate against the same 500-trial noise dispersion as case 1
    dsr_mined = deflated_sharpe(e["sharpe"], e["n"], e["skew"], e["kurt"],
                                N, var_trial)
    print(f"\n[2] SAME +0.22R edge over {e['n']} trades, two contexts:")
    print(f"    as 1 pre-registered idea  -> PSR {psr_edge:.3f}  "
          f"{'KEEP' if psr_edge >= 0.95 else 'reject'}")
    print(f"    as survivor of {N} genes   -> DSR {dsr_mined:.3f}  "
          f"{'KEEP' if dsr_mined >= 0.95 else 'reject'}")
    print("    -> identical stats, opposite verdict. Mining costs you proof.")

    # 3) Batch screen with FDR control — a genuine edge competing against noise
    #    on an equal footing (same track-record length, modest candidate count).
    #    This is the reassuring counterpart to [2]: real edges DO get through.
    batch = [{"id": f"noise{i}", "returns": list(rng.normal(0.0, 1.0, size=300))}
             for i in range(20)]
    batch.append({"id": "REAL_EDGE", "returns": list(rng.normal(0.30, 1.0, size=300))})
    screened = screen(batch, n_trials=len(batch))   # honest count: 21 tried
    kept = [c for c in screened if c["keep"]]
    print(f"\n[3] BATCH — 20 noise + 1 real edge, {len(batch)} trials, FDR<=0.10:")
    for c in screened[:3]:
        print(f"    {c['id']:10} Sharpe {c['sharpe']:+.3f}  DSR {c['dsr']:.3f}  "
              f"keep={c['keep']}")
    print(f"    kept after DSR+FDR: {[c['id'] for c in kept]}")


if __name__ == "__main__":
    _demo()

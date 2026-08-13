"""
SPRT — Wald's Sequential Probability Ratio Test for strategy expectancy
=======================================================================
Phase 2B, priority 3. Fixed thresholds (graduate at 15 trades, prove at 50) waste
samples: an obvious winner is still obvious at trade 8, an obvious loser at trade
6, yet both are forced to wait. SPRT replaces the fixed N with a sequential test
that stops as soon as the evidence is decisive AT a chosen error rate, and only
keeps sampling the genuinely ambiguous ones.

The test, on a stream of per-trade R-multiples x_i ~ N(mu, sigma^2):
    H0: mu = mu0   ("no edge" — retire)
    H1: mu = mu1   ("real edge" — graduate)
Gaussian log-likelihood ratio after n trades (sigma estimated from the sample):
    LLR_n = (mu1 - mu0)/sigma^2 * sum_i ( x_i - (mu0 + mu1)/2 )
Wald boundaries from the desired error rates:
    upper A = log((1 - beta) / alpha)        cross up  -> ACCEPT H1 (GRADUATE)
    lower B = log(beta / (1 - alpha))         cross down -> ACCEPT H0 (RETIRE)
    in between                                -> CONTINUE (need more trades)
  alpha = P(graduate | truly no edge)   = false-graduation rate
  beta  = P(retire   | truly has edge)  = false-retire rate

This is a drop-in decision function for the arena's graduate/relegate/retire
logic and for holdout_replay's gate. It does NOT replace the Deflated-Sharpe
guard — DSR corrects for how many strategies were tried (multiple testing); SPRT
decides WHEN one strategy's own record is conclusive. Use both: SPRT to stop
early, DSR to deflate by candidate count before granting live capital.

Self-test:  py sprt.py
"""

from __future__ import annotations
import sys, math
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Defaults tuned to the existing gate: "edge" means >= +0.10R expectancy (the
# arena's GRADUATE_MIN_EXP); "no edge" means breakeven-or-worse.
MU0_DEFAULT   = 0.0
MU1_DEFAULT   = 0.10
ALPHA_DEFAULT = 0.05      # tolerate a 5% chance of graduating a dud
BETA_DEFAULT  = 0.10      # tolerate a 10% chance of retiring a real edge
MIN_N         = 8         # never decide on fewer than this many trades
SIGMA_FLOOR   = 0.30      # floor on estimated R-std (avoids overconfidence early)
SIGMA_PRIOR   = 1.0       # used until enough trades to estimate sigma


def boundaries(alpha=ALPHA_DEFAULT, beta=BETA_DEFAULT):
    """(upper, lower) log-likelihood boundaries A, B."""
    A = math.log((1.0 - beta) / alpha)
    B = math.log(beta / (1.0 - alpha))
    return A, B


def llr(returns, mu0=MU0_DEFAULT, mu1=MU1_DEFAULT, sigma=None):
    """Gaussian sequential log-likelihood ratio for the whole stream so far."""
    r = np.asarray([x for x in returns if x is not None], dtype=float)
    n = len(r)
    if n == 0:
        return 0.0, 0, SIGMA_PRIOR
    if sigma is None:
        sigma = float(r.std(ddof=1)) if n >= 3 else SIGMA_PRIOR
    sigma = max(sigma, SIGMA_FLOOR)
    stat = (mu1 - mu0) / (sigma ** 2) * float(np.sum(r - (mu0 + mu1) / 2.0))
    return stat, n, sigma


def decide(returns, mu0=MU0_DEFAULT, mu1=MU1_DEFAULT,
           alpha=ALPHA_DEFAULT, beta=BETA_DEFAULT, min_n=MIN_N, sigma=None):
    """Sequential decision on a stream of R-multiples.
    Returns a dict: {n, expectancy, llr, upper, lower, sigma, decision} where
    decision in {GRADUATE, RETIRE, CONTINUE}. Stays CONTINUE below min_n."""
    stat, n, sig = llr(returns, mu0, mu1, sigma)
    A, B = boundaries(alpha, beta)
    exp = float(np.mean([x for x in returns if x is not None])) if n else 0.0
    if n < min_n:
        decision = "CONTINUE"
    elif stat >= A:
        decision = "GRADUATE"
    elif stat <= B:
        decision = "RETIRE"
    else:
        decision = "CONTINUE"
    return {"n": n, "expectancy": round(exp, 4), "llr": round(stat, 3),
            "upper": round(A, 3), "lower": round(B, 3),
            "sigma": round(sig, 3), "decision": decision}


def expected_n(mu_true, mu0=MU0_DEFAULT, mu1=MU1_DEFAULT,
               alpha=ALPHA_DEFAULT, beta=BETA_DEFAULT, sigma=1.0):
    """Wald's approximate expected number of trades to a decision when the true
    mean is mu_true. Handy for sizing the universe: smaller E[n] per strategy =
    fewer parallel trades needed before a verdict."""
    A, B = boundaries(alpha, beta)
    # per-trade expected LLR increment under mu_true
    incr = (mu1 - mu0) / (sigma ** 2) * (mu_true - (mu0 + mu1) / 2.0)
    if abs(incr) < 1e-9:
        return float("inf")
    if mu_true >= (mu0 + mu1) / 2.0:        # likely to accept H1
        p_h1 = 1.0 - beta if mu_true >= mu1 else 0.5
        return (p_h1 * A + (1 - p_h1) * B) / incr
    p_h1 = alpha if mu_true <= mu0 else 0.5
    return (p_h1 * A + (1 - p_h1) * B) / incr


# ─── SELF-TEST ────────────────────────────────────────────────────────────────

def _selftest():
    rng = np.random.default_rng(11)
    A, B = boundaries()
    print("=" * 64)
    print(f"SPRT self-test  (mu0={MU0_DEFAULT} mu1={MU1_DEFAULT} "
          f"alpha={ALPHA_DEFAULT} beta={BETA_DEFAULT})")
    print(f"  boundaries: GRADUATE if LLR>={A:.2f}, RETIRE if LLR<={B:.2f}")
    print("=" * 64)

    def run(label, mu, sd=1.0, cap=400):
        # feed trades one at a time; report when (if) it decides
        stream = []
        for k in range(1, cap + 1):
            stream.append(float(rng.normal(mu, sd)))
            d = decide(stream)
            if d["decision"] != "CONTINUE":
                print(f"  {label:22} mu={mu:+.2f}: {d['decision']:8} "
                      f"at n={d['n']:3}  (exp {d['expectancy']:+.3f}R, LLR {d['llr']:+.2f})")
                return d["decision"], d["n"]
        print(f"  {label:22} mu={mu:+.2f}: no decision in {cap} (exp "
              f"{np.mean(stream):+.3f}R)")
        return "CONTINUE", cap

    g, ng = run("clear winner", 0.25)
    l, nl = run("clear loser", -0.15)
    run("marginal edge", 0.10)
    run("pure noise", 0.0)
    assert g == "GRADUATE", "a +0.25R edge must graduate"
    assert l == "RETIRE", "a -0.15R stream must retire"
    print(f"\n  E[n] winner(+0.25R) ~ {expected_n(0.25):.0f}, "
          f"loser(-0.15R) ~ {expected_n(-0.15):.0f} trades "
          f"(vs fixed 15/50) -> fewer trades to a verdict.")
    print("  PASS")


if __name__ == "__main__":
    _selftest()

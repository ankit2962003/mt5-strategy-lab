"""
Monte Carlo — risk-of-ruin & prop-challenge odds from a strategy's trade record
===============================================================================
Your top-interest item, built to answer the questions that actually matter for a
$10k funded challenge — not "what was the backtest expectancy" but "across
thousands of plausible re-orderings of these trades, how often do I PASS, and how
often do I BLOW THE DRAWDOWN LIMIT first?"

It resamples a strategy's per-trade R-multiples (BLOCK bootstrap, so losing
streaks stay clustered — that's what actually causes ruin) and walks each path
with the SAME drawdown-based sizing the live runner uses (risk_guard.dd_risk_usd:
5% of the remaining $1000 buffer, $50 at full, shrinking as you draw down). For
each path it checks whether you hit the +$800 target or the −$1000 drawdown
limit first.

Outputs per strategy:
  - P(pass)      : reach +8% ($800) before −10% drawdown ($1000)
  - P(ruin)      : hit the drawdown limit first
  - P(profit)    : positive over a fixed horizon
  - drawdown distribution (median / 95th-percentile worst drawdown)
  - equity percentiles (5th / 50th / 95th)

HONESTY KNOB: the field record is flat-spread (optimistic). `extra_cost_r`
subtracts a per-trade cost so you can stress the odds toward real-tick reality
(we measured bar→tick costing ~0.1–0.15R on gold). Run it with a haircut.

    py monte_carlo.py selftest
    py monte_carlo.py             # field table, ranked by P(pass), writes monte_carlo.json
    py monte_carlo.py SID         # one strategy in detail (+ plots/CSV to output/mc_plots/)
    py monte_carlo.py SID 0.1     # ...stress-tested with a 0.1R cost haircut
    py monte_carlo.py plotdemo    # synthetic run that writes the viz artefacts (smoke test)

VISUALISATION STEP (additional, console summary unchanged): after running N paths,
visualize()/save_visuals() write to output/mc_plots/ at 150dpi (Agg backend):
  equity_fan.png · terminal_equity_hist.png · max_drawdown_hist.png · mc_paths.csv
"""

from __future__ import annotations
import sys, os, json
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# reuse the existing data plumbing so MC sees the same records as DSR/Kelly
from self_improve import _forward_by_sid, _load, TRACK_PATH, GRADUATED_PATH

BASE     = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(BASE, "monte_carlo.json")

# ── prop-challenge parameters (a $10k funded account) ────────────────────────
DD_LIMIT      = 1000.0    # −10% : ruin (overall max drawdown, mirrors risk_guard.MAX_DRAWDOWN_LIMIT)
DAILY_DD_LIMIT= 500.0     # −5%  : prop-firm daily loss limit (risk_guard.DAILY_LOSS_LIMIT) — reference line only
PROFIT_TARGET = 800.0     # +8%  : pass
DD_FRACTION   = 0.05      # drawdown-based sizing: 5% of remaining buffer
RISK_CAP      = 50.0
RISK_FLOOR    = 10.0
N_SIMS        = 10_000
HORIZON       = 200       # trades simulated for the fixed-horizon distribution
BLOCK         = 5         # block-bootstrap length (preserves streaks)
MIN_TRADES    = 20        # need at least this many real trades to simulate


# ── sizing models (mirror risk_guard) ────────────────────────────────────────

def dd_risk(cum_pnl):
    """Drawdown-based $ risk for the next trade given current cumulative P&L."""
    used = max(0.0, -cum_pnl)
    buffer = max(0.0, DD_LIMIT - used)
    return max(RISK_FLOOR, min(RISK_CAP, DD_FRACTION * buffer))

def fixed_risk(_cum):
    return RISK_CAP


# ── resampling ────────────────────────────────────────────────────────────────

def block_bootstrap(rs, n, block, rng):
    """Draw `n` R-multiples by stitching random contiguous blocks (with wrap), so
    serial structure (winning/losing streaks) survives the resample."""
    rs = np.asarray(rs, float)
    L = len(rs)
    if L == 0:
        return np.zeros(n)
    out = np.empty(n)
    i = 0
    while i < n:
        s = int(rng.integers(0, L))
        take = min(block, n - i)
        idx = (s + np.arange(take)) % L
        out[i:i + take] = rs[idx]
        i += take
    return out


# ── single-path simulators ────────────────────────────────────────────────────

def _path_challenge(rs, rng, sizing, extra_cost_r, max_trades, block):
    """Walk one resampled path until PASS (+target), RUIN (−DD limit), or timeout.
    Returns (outcome, final_pnl, max_drawdown, n_trades)."""
    seq = block_bootstrap(rs, max_trades, block, rng) - extra_cost_r
    cum = 0.0; peak = 0.0; maxdd = 0.0
    for k, R in enumerate(seq, 1):
        cum += sizing(cum) * R
        peak = max(peak, cum)
        maxdd = max(maxdd, peak - cum)
        if cum <= -DD_LIMIT:
            return "ruin", cum, maxdd, k
        if cum >= PROFIT_TARGET:
            return "pass", cum, maxdd, k
    return "timeout", cum, maxdd, max_trades

def _path_horizon(rs, rng, sizing, extra_cost_r, horizon, block):
    """Walk a fixed `horizon` of trades (no early stop). Returns (final, maxdd,
    touched_ruin)."""
    seq = block_bootstrap(rs, horizon, block, rng) - extra_cost_r
    cum = 0.0; peak = 0.0; maxdd = 0.0; ruin = False
    for R in seq:
        cum += sizing(cum) * R
        peak = max(peak, cum)
        maxdd = max(maxdd, peak - cum)
        if cum <= -DD_LIMIT:
            ruin = True
    return cum, maxdd, ruin

def _path_horizon_full(rs, rng, sizing, extra_cost_r, horizon, block):
    """Like _path_horizon but also returns the full equity curve (length
    horizon+1, starting at 0) and the longest run of consecutive losing trades.
    Used only for the visualisation step — the summary stats above are untouched."""
    seq = block_bootstrap(rs, horizon, block, rng) - extra_cost_r
    curve = np.empty(horizon + 1)
    curve[0] = 0.0
    cum = 0.0; peak = 0.0; maxdd = 0.0
    consec = 0; max_consec = 0
    for i, R in enumerate(seq, 1):
        cum += sizing(cum) * R
        curve[i] = cum
        peak = max(peak, cum)
        maxdd = max(maxdd, peak - cum)
        if R < 0:
            consec += 1
            max_consec = max(max_consec, consec)
        else:
            consec = 0
    return curve, cum, maxdd, max_consec


# ── top-level Monte Carlo over a strategy's R-record ──────────────────────────

def simulate(rs, n_sims=N_SIMS, horizon=HORIZON, block=BLOCK, extra_cost_r=0.0,
             sizing="dd", seed=12345):
    """Full Monte Carlo on a per-trade R-multiple record. Returns the risk/odds
    dict. `sizing` = 'dd' (drawdown-based, default) or 'fixed' ($50/trade)."""
    rs = [float(x) for x in rs if x is not None]
    if len(rs) < 2:
        return {"error": "too few trades", "n": len(rs)}
    rng = np.random.default_rng(seed)
    size_fn = dd_risk if sizing == "dd" else fixed_risk

    # challenge odds (early-stop on pass/ruin)
    outcomes = {"pass": 0, "ruin": 0, "timeout": 0}
    trades_to_end = []
    for _ in range(n_sims):
        out, _f, _dd, k = _path_challenge(rs, rng, size_fn, extra_cost_r,
                                          max_trades=horizon, block=block)
        outcomes[out] += 1
        trades_to_end.append(k)

    # fixed-horizon outcome distribution
    finals = np.empty(n_sims); maxdds = np.empty(n_sims); ruins = 0
    for j in range(n_sims):
        f, dd, ru = _path_horizon(rs, rng, size_fn, extra_cost_r, horizon, block)
        finals[j] = f; maxdds[j] = dd; ruins += int(ru)

    return {
        "n_trades_in_record": len(rs),
        "expectancy_R": round(float(np.mean(rs)) - extra_cost_r, 4),
        "extra_cost_r": extra_cost_r, "sizing": sizing, "horizon": horizon,
        # challenge (the headline)
        "p_pass":   round(outcomes["pass"] / n_sims, 4),
        "p_ruin":   round(outcomes["ruin"] / n_sims, 4),
        "p_timeout":round(outcomes["timeout"] / n_sims, 4),
        "median_trades_to_resolve": int(np.median(trades_to_end)),
        # fixed-horizon risk
        "p_profit":        round(float((finals > 0).mean()), 4),
        "risk_of_ruin":    round(ruins / n_sims, 4),
        "final_p05":  round(float(np.quantile(finals, 0.05)), 2),
        "final_p50":  round(float(np.quantile(finals, 0.50)), 2),
        "final_p95":  round(float(np.quantile(finals, 0.95)), 2),
        "maxdd_median": round(float(np.quantile(maxdds, 0.50)), 2),
        "maxdd_p95":    round(float(np.quantile(maxdds, 0.95)), 2),
    }


# ── visualisation (additional step; summary stats above are unchanged) ────────

PLOT_DIR = os.path.join(BASE, "output", "mc_plots")   # "/output/mc_plots/" under the repo


def simulate_paths(rs, n_sims=N_SIMS, horizon=HORIZON, block=BLOCK,
                   extra_cost_r=0.0, sizing="dd", seed=12345):
    """Run n_sims fixed-horizon paths and KEEP the raw per-path data so we can
    plot/persist it. Returns equity matrix (n_sims, horizon+1) plus per-path
    terminal equity, max drawdown, and max consecutive losses."""
    rs = [float(x) for x in rs if x is not None]
    if len(rs) < 2:
        return None
    rng = np.random.default_rng(seed)
    size_fn = dd_risk if sizing == "dd" else fixed_risk
    equity   = np.empty((n_sims, horizon + 1))
    finals   = np.empty(n_sims)
    maxdds   = np.empty(n_sims)
    maxcons  = np.empty(n_sims, dtype=int)
    for j in range(n_sims):
        curve, f, dd, mc = _path_horizon_full(rs, rng, size_fn, extra_cost_r,
                                              horizon, block)
        equity[j] = curve; finals[j] = f; maxdds[j] = dd; maxcons[j] = mc
    return {"equity": equity, "finals": finals, "maxdds": maxdds,
            "max_consec_losses": maxcons, "horizon": horizon, "n_sims": n_sims}


def save_visuals(paths, out_dir=PLOT_DIR, dpi=150, max_lines=400, label=""):
    """Write equity_fan.png, terminal_equity_hist.png, max_drawdown_hist.png and
    mc_paths.csv to out_dir. `paths` is the dict from simulate_paths()."""
    import matplotlib
    matplotlib.use("Agg")                 # non-interactive backend
    import matplotlib.pyplot as plt
    import csv

    os.makedirs(out_dir, exist_ok=True)
    eq      = paths["equity"]
    finals  = paths["finals"]
    maxdds  = paths["maxdds"]
    maxcons = paths["max_consec_losses"]
    n, steps = eq.shape
    x = np.arange(steps)
    tag = f" — {label}" if label else ""

    # per-trade-index percentiles across ALL paths (band uses every path)
    p05 = np.quantile(eq, 0.05, axis=0)
    p50 = np.quantile(eq, 0.50, axis=0)
    p95 = np.quantile(eq, 0.95, axis=0)
    # a real path to highlight: the one whose terminal is closest to the median
    med_idx = int(np.argmin(np.abs(finals - np.median(finals))))

    # ── 1. equity fan ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 6))
    stride = max(1, n // max_lines)        # cap the overlaid lines for legibility
    ax.plot(x, eq[::stride].T, color="steelblue", lw=0.5, alpha=0.1)
    ax.fill_between(x, p05, p95, color="orange", alpha=0.25,
                    label="5th–95th pct band")
    ax.plot(x, p50, color="darkorange", lw=2.0, label="median path (pct)")
    ax.plot(x, eq[med_idx], color="crimson", lw=1.6, ls="--",
            label="representative median path")
    ax.axhline(PROFIT_TARGET, color="green", lw=1.0, ls=":",
               label=f"+${PROFIT_TARGET:.0f} target")
    ax.axhline(-DD_LIMIT, color="red", lw=1.0, ls=":",
               label=f"-${DD_LIMIT:.0f} DD limit")
    ax.set_title(f"Monte Carlo equity fan — {n} paths × {steps-1} trades{tag}")
    ax.set_xlabel("trade #"); ax.set_ylabel("equity P&L ($)")
    ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "equity_fan.png"), dpi=dpi)
    plt.close(fig)

    # ── 2. terminal equity histogram + VaR/CVaR ──────────────────────────────
    var95  = float(np.quantile(finals, 0.05))      # 95% VaR: 5th-pctile outcome
    cvar95 = float(finals[finals <= var95].mean()) # mean of the worst 5% tail
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(finals, bins=60, color="steelblue", alpha=0.8)
    ax.axvline(var95, color="darkorange", lw=2.0,
               label=f"VaR(95%) = ${var95:,.0f}")
    ax.axvline(cvar95, color="crimson", lw=2.0,
               label=f"CVaR(95%) = ${cvar95:,.0f}")
    ax.axvline(0, color="gray", lw=1.0, ls=":")
    ax.set_title(f"Terminal equity distribution — {n} paths{tag}")
    ax.set_xlabel("terminal equity P&L ($)"); ax.set_ylabel("paths")
    ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "terminal_equity_hist.png"), dpi=dpi)
    plt.close(fig)

    # ── 3. max-drawdown histogram + prop-firm DD limits ──────────────────────
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(maxdds, bins=60, color="indianred", alpha=0.8)
    ax.axvline(DAILY_DD_LIMIT, color="orange", lw=2.0,
               label=f"daily limit ${DAILY_DD_LIMIT:.0f}")
    ax.axvline(DD_LIMIT, color="darkred", lw=2.0,
               label=f"overall limit ${DD_LIMIT:.0f}")
    ax.set_title(f"Max drawdown per path — {n} paths{tag}")
    ax.set_xlabel("max drawdown ($)"); ax.set_ylabel("paths")
    ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "max_drawdown_hist.png"), dpi=dpi)
    plt.close(fig)

    # ── 4. raw per-path data (re-plottable without rerunning) ─────────────────
    csv_path = os.path.join(out_dir, "mc_paths.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "terminal_equity", "max_drawdown",
                    "max_consecutive_losses"])
        for i in range(n):
            w.writerow([i, round(float(finals[i]), 2),
                        round(float(maxdds[i]), 2), int(maxcons[i])])

    return {"out_dir": out_dir, "var95": round(var95, 2),
            "cvar95": round(cvar95, 2),
            "files": ["equity_fan.png", "terminal_equity_hist.png",
                      "max_drawdown_hist.png", "mc_paths.csv"]}


def visualize(rs, out_dir=PLOT_DIR, n_sims=N_SIMS, extra_cost_r=0.0,
              sizing="dd", seed=12345, label=""):
    """Convenience: run paths + save all four artefacts in one call."""
    paths = simulate_paths(rs, n_sims=n_sims, extra_cost_r=extra_cost_r,
                           sizing=sizing, seed=seed)
    if paths is None:
        return {"error": "too few trades"}
    return save_visuals(paths, out_dir=out_dir, label=label)


# ── field runner (all strategies) ─────────────────────────────────────────────

def run_all(extra_cost_r=0.0, write=True, n_sims=4000):
    fwd   = _forward_by_sid()
    track = _load(TRACK_PATH, {})
    grad  = set(_load(GRADUATED_PATH, []))
    out = {}
    for sid in set(fwd) | set(track):
        rs = list(track.get(sid, {}).get("backtest_rs", [])) + list(fwd.get(sid, []))
        if len(rs) < MIN_TRADES:
            continue
        r = simulate(rs, n_sims=n_sims, extra_cost_r=extra_cost_r)
        if "error" in r:
            continue
        r["symbol"] = track.get(sid, {}).get("symbol", "")
        r["live"] = sid in grad
        out[sid] = r
    if write:
        json.dump(out, open(OUT_PATH, "w"), indent=2)
    return out


def _print_table(res):
    rows = [dict(v, sid=k) for k, v in res.items()]
    rows.sort(key=lambda r: r["p_pass"], reverse=True)
    print(f"  {'sid':10} {'sym':7} {'exp':>7} {'P(pass)':>8} {'P(ruin)':>8} "
          f"{'P(profit)':>9} {'maxDD95':>8} {'live':>4}")
    for r in rows[:25]:
        print(f"  {r['sid']:10} {r['symbol']:7} {r['expectancy_R']:+7.3f} "
              f"{r['p_pass']*100:7.1f}% {r['p_ruin']*100:7.1f}% "
              f"{r['p_profit']*100:8.1f}% ${r['maxdd_p95']:7.0f} {'Y' if r['live'] else '':>4}")


# ── SELF-TEST ─────────────────────────────────────────────────────────────────

def _selftest():
    rng = np.random.default_rng(1)
    print("=" * 70)
    print("Monte Carlo self-test (dd-based sizing, $1000 DD limit / $800 target)")
    print("=" * 70)
    cases = [
        ("strong edge +0.30R", list(rng.normal(0.30, 1.0, 300))),
        ("small edge  +0.10R", list(rng.normal(0.10, 1.0, 300))),
        ("breakeven    0.00R", list(rng.normal(0.00, 1.0, 300))),
        ("loser       -0.10R", list(rng.normal(-0.10, 1.0, 300))),
    ]
    prev_pass = 1.1
    for label, rs in cases:
        r = simulate(rs, n_sims=4000)
        print(f"\n  {label}:")
        print(f"     P(pass) {r['p_pass']*100:5.1f}%  P(ruin) {r['p_ruin']*100:5.1f}%  "
              f"P(profit@{r['horizon']}) {r['p_profit']*100:5.1f}%")
        print(f"     final $ p05/p50/p95 = {r['final_p05']:+.0f}/{r['final_p50']:+.0f}/"
              f"{r['final_p95']:+.0f}   worst DD (p95) ${r['maxdd_p95']:.0f}")
        assert r["p_pass"] <= prev_pass + 0.02, "P(pass) must fall as edge weakens"
        prev_pass = r["p_pass"]
    # monotonic sanity: strong edge passes more than it ruins; loser the reverse
    strong = simulate(cases[0][1], n_sims=4000)
    loser  = simulate(cases[3][1], n_sims=4000)
    assert strong["p_pass"] > strong["p_ruin"], "strong edge should pass > ruin"
    assert loser["p_ruin"] > loser["p_pass"], "loser should ruin > pass"
    # cost haircut must lower the odds
    base = simulate(cases[1][1], n_sims=4000, extra_cost_r=0.0)
    hair = simulate(cases[1][1], n_sims=4000, extra_cost_r=0.15)
    print(f"\n  cost haircut 0.15R on the small edge: P(pass) "
          f"{base['p_pass']*100:.1f}% -> {hair['p_pass']*100:.1f}%  (must drop)")
    assert hair["p_pass"] < base["p_pass"], "a cost haircut must lower P(pass)"
    print("\n  PASS — odds fall with edge, ruin dominates losers, costs lower the odds.")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] == "run":
        cost = float(args[1]) if len(args) > 1 else 0.0
        res = run_all(extra_cost_r=cost)
        print(f"Monte Carlo — {len(res)} strategies (>= {MIN_TRADES} trades), "
              f"cost haircut {cost}R, written to monte_carlo.json\n")
        _print_table(res)
    elif args[0] == "selftest":
        _selftest()
    elif args[0] == "plotdemo":
        # synthetic small-edge record — verify the viz step end-to-end without live data
        rng = np.random.default_rng(7)
        rs = list(rng.normal(0.10, 1.0, 300))
        info = visualize(rs, n_sims=4000, label="demo +0.10R")
        print(f"plotdemo: wrote {', '.join(info['files'])} to {info['out_dir']}")
        print(f"          VaR(95%)=${info['var95']:,.0f}  CVaR(95%)=${info['cvar95']:,.0f}")
    else:
        sid = args[0]
        cost = float(args[1]) if len(args) > 1 else 0.0
        track = _load(TRACK_PATH, {}); fwd = _forward_by_sid()
        rs = list(track.get(sid, {}).get("backtest_rs", [])) + list(fwd.get(sid, []))
        if len(rs) < MIN_TRADES:
            print(f"{sid}: only {len(rs)} trades (<{MIN_TRADES}) — too few to simulate")
        else:
            print(json.dumps({sid: simulate(rs, extra_cost_r=cost)}, indent=2))
            # additional step: per-path plots + raw CSV (console summary above unchanged)
            info = visualize(rs, extra_cost_r=cost, label=sid)
            print(f"\nplots+csv -> {info['out_dir']}  "
                  f"({', '.join(info['files'])})")

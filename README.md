# mt5-strategy-lab

**Validation infrastructure for systematic trading research** — the machinery that decides
whether a strategy result is real, and the execution realism that stops a backtest from
flattering itself.

> **Scope note:** this repository publishes the *validation and execution infrastructure*,
> deliberately **not** the strategy logic (signal generation, evolved rule sets, or the
> live research pipeline). The interesting engineering here is the guardrails, not the
> signals. Some modules therefore reference strategy-side helpers that are not included —
> see [Not included](#not-included).

---

## The problem this exists to solve

Generate enough candidate strategies and some will clear any threshold by luck alone. A
single 70/30 split gives you one out-of-sample path and one number — easy to get lucky on.
And a clean-OHLC backtest is fiction that dies on live spread: a gold strategy that showed
**+0.47R on paper bled negative live purely on spread**.

Every module here exists because of a specific way that research goes wrong.

## `validation/` — not fooling yourself

| Module | What it does | Failure mode it kills |
|---|---|---|
| **`stat_guard.py`** | Deflated Sharpe Ratio + Benjamini–Hochberg FDR control | Multiple testing. Test 1,000 evolved genes and the best one looks great by construction. DSR asks whether the winner beats what the *best of N random tries* would have produced. |
| **`cpcv.py`** | Combinatorial Purged Cross-Validation (López de Prado) | Single-split luck. Splits bars into N groups, holds out k in **every** combination, and purges/embargoes around the boundary so leakage across adjacent bars can't inflate the score. |
| **`holdout.py`** | A walled-off, **rotating** window the search never trains on | Contamination. If graduation is decided on data the optimiser already saw, every "winner" is compromised. |
| **`sprt.py`** | Wald's Sequential Probability Ratio Test on expectancy | Wasted samples. Fixed thresholds ("graduate at 15 trades") make an obvious loser wait; SPRT stops as soon as the evidence is decisive, in either direction. |

## `execution/` — not flattering yourself

| Module | What it does |
|---|---|
| **`tick_replay.py`** | Backtests on **real MT5 tick data**, paying the actual bid/ask spread and slippage on every fill rather than assuming mid-price execution. |
| **`monte_carlo.py`** | Risk-of-ruin and funded-challenge pass odds by resampling a strategy's real trade record thousands of times — answering "how often do I survive?", not "what was the backtest expectancy?" |

---

## Design principles

- **A method that can only confirm you isn't a method.** Every component here can return a
  negative verdict, and negative verdicts are the point.
- **Pessimistic by default.** Costs, spread and slippage are charged against you; where an
  assumption is uncertain, the unfavourable one is used.
- **Purge and embargo, always.** Adjacent bars leak. Cross-validation that ignores this
  reports a number that cannot survive contact with live data.
- **State the sample size.** An underpowered result is reported as underpowered, not
  rounded up into a conclusion.

## Not included

These modules are extracted from a larger private research monorepo and reference
strategy-side helpers that are intentionally withheld:

- `strategy_lab` — signal predicates and take-profit logic (`cpcv.py`, `tick_replay.py`)
- `self_improve` — the live track record store (`monte_carlo.py`)

`stat_guard.py`, `holdout.py` and `sprt.py` are self-contained and depend only on NumPy.
The others will raise `ImportError` on a bare clone — they are published to show the
approach, not as a turnkey install.

## Requirements

Python 3.10+, NumPy. `pandas`/`MetaTrader5` for tick replay (Windows), `matplotlib`
optionally for Monte Carlo plots. See [`requirements.txt`](requirements.txt).

## Licence

See [LICENSE](LICENSE).

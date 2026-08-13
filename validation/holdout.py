"""
Holdout — a walled-off, ROTATING window the GA never trains on
==============================================================
Phase 2B, priority 2 (the guardrail half). The lab's genetic search is a
multiple-testing machine; if graduation is decided on data the GA already saw,
every "winner" is contaminated. So we reserve a recent window that the lab is
forbidden to train on, and grade graduation candidates by replaying them there
on real ticks (holdout_replay.py).

Two guardrails live here, both mandatory per the Phase 2B brief:

  1. CANDIDATE COUNT -> DSR.  Every distinct strategy graded against the current
     holdout is one more lottery ticket. We count them (deduped by sid) and feed
     that count into the Deflated-Sharpe gate, so testing more candidates raises
     the bar instead of blessing noise.

  2. ROTATION.  Repeatedly grading on the SAME frozen window slowly overfits it
     (you keep the strategies that happen to fit those weeks). So the window
     rotates forward on a schedule; when it does, the candidate counter resets —
     a fresh, never-graded window starts clean.

Window convention: the holdout is the most-recent `HOLDOUT_DAYS` of calendar
time, ending "now". Training data is everything STRICTLY BEFORE `cutoff_ms`.
Honest constraint (2026-06-21): the demo only serves ~30-35 days of real ticks,
so HOLDOUT_DAYS is sized to what real ticks can actually cover today; the window
deepens as tick_replay.archive_forward() hoards ticks going forward.

State: holdout_state.json
  { "cutoff_ms", "end_ms", "rotated_utc", "candidates": {sid: first_tested_utc},
    "history": [ {window, n_candidates, retired_utc} ... ] }
"""

import os, json
from datetime import datetime, timezone, timedelta

BASE        = os.path.dirname(os.path.abspath(__file__))
STATE_PATH  = os.path.join(BASE, "holdout_state.json")

# Window length. Kept within the real-tick horizon (~30d) so the holdout can be
# graded on REAL ticks today; raise as the forward archive deepens. The lab does
# NOT pay a data-depth cost for this — it simply shifts its training window to end
# at the cutoff (strategy_lab.get_bars_until), so a longer holdout just trains on
# slightly older bars, it doesn't shrink the training sample.
HOLDOUT_DAYS = 21
# Rotate the window forward this often. Shorter = less overfit per window but a
# shorter graded record; 10d balances the two given one-at-a-time throughput.
ROTATE_DAYS  = 10


def _now():
    return datetime.now(timezone.utc)

def _ms(dt):
    return int(dt.timestamp() * 1000)

def _load():
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return None

def _save(st):
    with open(STATE_PATH, "w") as f:
        json.dump(st, f, indent=2)


def _fresh_window(now=None):
    now = now or _now()
    end = now
    cutoff = end - timedelta(days=HOLDOUT_DAYS)
    return {
        "cutoff_ms": _ms(cutoff),
        "end_ms": _ms(end),
        "rotated_utc": now.isoformat(timespec="seconds"),
        "candidates": {},      # sid -> first-tested ISO (dedup so re-tests don't inflate)
        "history": [],
    }


def state(auto_rotate=True):
    """Return the current holdout state, initializing on first use and rotating
    forward if the window is past due. Rotation archives the spent window's
    candidate count and starts a clean one."""
    st = _load()
    now = _now()
    if st is None:
        st = _fresh_window(now)
        _save(st)
        return st
    if auto_rotate:
        rotated = datetime.fromisoformat(st["rotated_utc"])
        if (now - rotated) >= timedelta(days=ROTATE_DAYS):
            st.setdefault("history", []).append({
                "cutoff_ms": st["cutoff_ms"], "end_ms": st["end_ms"],
                "rotated_utc": st["rotated_utc"],
                "n_candidates": len(st.get("candidates", {})),
                "retired_utc": now.isoformat(timespec="seconds"),
            })
            new = _fresh_window(now)
            new["history"] = st["history"][-20:]
            st = new
            _save(st)
    return st


def current_window(auto_rotate=True):
    """(cutoff_ms, end_ms): graduation candidates are graded on [cutoff_ms, end_ms];
    the GA must train only on bars with open time < cutoff_ms."""
    st = state(auto_rotate=auto_rotate)
    return st["cutoff_ms"], st["end_ms"]


def is_holdout_ms(ts_ms):
    """True if an epoch-ms timestamp falls inside the current holdout window."""
    c, e = current_window()
    return c <= ts_ms <= e


def exclude_holdout(df, epoch_s):
    """Trim a prepared bar frame to TRAINING data only (open time < cutoff). The
    lab calls this so the GA never sees the held-out window. `epoch_s` is the
    per-bar open time in epoch seconds (parallel to df). Returns a boolean mask
    AND the trimmed positions so callers can slice df + any parallel arrays."""
    cutoff_ms, _ = current_window()
    mask = (epoch_s.astype("int64") * 1000) < cutoff_ms
    return mask


def register_candidate(sid):
    """Record that `sid` was graded against the current holdout. Deduped: a sid
    re-tested within the same window does NOT increase the count (you don't get
    punished for re-checking the same idea, only for trying NEW ones). Returns
    the post-registration candidate count to feed into DSR."""
    st = state()
    cands = st.setdefault("candidates", {})
    if sid not in cands:
        cands[sid] = _now().isoformat(timespec="seconds")
        _save(st)
    return len(cands)


def n_candidates():
    """How many distinct strategies have been graded on the current window. This
    is the n_trials handed to the Deflated-Sharpe gate."""
    return len(state().get("candidates", {}))


def info():
    st = state(auto_rotate=False)
    c, e = st["cutoff_ms"], st["end_ms"]
    cd = datetime.fromtimestamp(c / 1000, timezone.utc).date()
    ed = datetime.fromtimestamp(e / 1000, timezone.utc).date()
    days_in = (datetime.fromisoformat(st["rotated_utc"]) +
               timedelta(days=ROTATE_DAYS) - _now()).days
    return (f"Holdout window: {cd} .. {ed}  ({HOLDOUT_DAYS}d)\n"
            f"  candidates graded this window: {len(st.get('candidates', {}))}\n"
            f"  rotates in ~{max(0, days_in)}d (every {ROTATE_DAYS}d)\n"
            f"  past windows archived: {len(st.get('history', []))}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "rotate":
        # force a rotation (testing / manual)
        st = state(auto_rotate=False)
        st["rotated_utc"] = (_now() - timedelta(days=ROTATE_DAYS + 1)).isoformat()
        _save(st)
        state()  # triggers rotation
        print("Forced rotation.\n")
    print(info())

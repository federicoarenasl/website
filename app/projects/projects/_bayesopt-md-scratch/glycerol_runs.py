"""Locate the run directories to plot, ignoring dead ones.

Every agent run writes to a fresh timestamped directory under
``runs/glycerol/<arm>/run_<seed>/``. An arm that was launched, failed and
relaunched therefore leaves several, and picking the wrong one is silent: the
figure renders, it just plots a run that died after one step. That happened --
the LLM arm's first two launches (a credit exhaustion, and an unpicklable SDK
exception) left 1-step and 0-step directories that sorted *before* the real
30-step run, so ``sorted(...)[0]`` selected a crashed attempt.

Selection is therefore by step count first, recency second: the longest run
for a seed, breaking ties toward the most recent.
"""

from pathlib import Path

REPO = Path("/Users/federico/Documents/personal/code/agentic-optimiser")


def _rows(path: Path) -> int:
    try:
        with path.open() as f:
            return sum(1 for _ in f) - 1
    except OSError:
        return 0


def _budget(arm: str) -> int:
    """Longest run this arm produced -- taken as its intended step budget."""
    return max((_rows(p) for p in
                (REPO / "runs" / "glycerol" / arm).glob("run_*/*/steps.csv")),
               default=0)


def steps_csv(arm: str, seed: int, min_fraction: float = 0.8) -> Path | None:
    """The most complete steps.csv for one arm and seed, or None.

    A run that stopped after one or two steps is a failure, not a shorter
    experiment, so runs below ``min_fraction`` of the arm's budget are
    discarded rather than plotted. The fraction rather than an exact match
    leaves room for a run that ended early on the stall guard.
    """
    floor = max(2, int(min_fraction * _budget(arm)))
    candidates = list((REPO / "runs" / "glycerol" / arm).glob(f"run_{seed}/*/steps.csv"))
    good = [c for c in candidates if _rows(c) >= floor]
    if not good:
        return None
    return max(good, key=lambda p: (_rows(p), p.name))


def all_seeds(arm: str, max_seeds: int = 10) -> list[Path]:
    """One steps.csv per seed that has a usable run, in seed order."""
    found = [steps_csv(arm, s) for s in range(max_seeds)]
    return [p for p in found if p is not None]


def representative_seed(arm: str) -> int:
    """The seed whose best loss is closest to the arm's median.

    Figures show one seed. Defaulting to seed 0 is a silent editorial choice:
    here it happened to be the LLM arm's exact median but 1.8x worse than
    typical for the surrogate, which would have made a figure that flattered
    the conclusion. Picking by distance from the median removes the choice from
    whoever writes the script.
    """
    import csv
    import statistics as st

    best = {}
    for seed in range(10):
        path = steps_csv(arm, seed)
        if path is None:
            continue
        with path.open() as f:
            losses = [float(r["loss"]) for r in csv.DictReader(f) if r["loss"]]
        if losses:
            best[seed] = min(losses)
    if not best:
        return 0
    median = st.median(best.values())
    return min(best, key=lambda s: abs(best[s] - median))

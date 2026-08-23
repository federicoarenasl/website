"""Did the arms find good physics, or one lucky trajectory?

The reported best loss is the minimum over many noisy measurements, so it is
biased low: the more simulations an arm runs, the more chances it has to draw a
favourable trajectory. This re-runs each seed's best parameters on a *fresh*
seed and reports the honest loss.

An arm whose advantage was real holds its value. An arm that was fitting noise
regresses.

    cd /Users/federico/Documents/personal/code/agentic-optimiser
    MD_ENGINE=argon ./.venv/bin/python <this file>
"""

import csv
import glob
import statistics
import sys
from pathlib import Path

REPO = Path("/Users/federico/Documents/personal/code/agentic-optimiser")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))

from optimiser.md import ARGON_EPSILON, ARGON_SIGMA  # noqa: E402
from optimiser.simulator import PARAM_NAMES, run_md_simulation  # noqa: E402

from generate_data import OUT  # noqa: E402

RUNS_DIR = REPO / "runs" / "argon"
VALIDATION_SEED_OFFSET = 1000  # seeds never used during optimisation

ARMS = [
    ("Random search", "random"),
    ("GP (untuned)", "gp_untuned"),
    ("GP (tuned), random warm-up", "gp_tuned"),
    ("GP (tuned), LLM warm-up", "gp_llm_warm"),
    ("LLM only (no surrogate)", "llm_only"),
]


def best_points(tag: str) -> list[tuple[int, dict, float]]:
    """The lowest-loss parameter set from each seed of one arm."""
    found = []
    for seed_dir in sorted(RUNS_DIR.glob(f"{tag}/run_*")):
        seed = int(seed_dir.name.split("_")[1])
        steps = glob.glob(str(seed_dir / "*" / "steps.csv"))
        if not steps:
            continue
        rows = [r for r in csv.DictReader(open(steps[0])) if r["loss"] not in ("", None)]
        if not rows:
            continue
        best = min(rows, key=lambda r: float(r["loss"]))
        found.append((seed, {k: float(best[k]) for k in PARAM_NAMES}, float(best["loss"])))
    return found


def main() -> None:
    print("Re-running each seed's best parameters on unseen seeds.\n")
    rows = []
    for label, tag in ARMS:
        points = best_points(tag)
        reported, revalidated = [], []
        for seed, params, loss in points:
            reported.append(loss)
            fresh = run_md_simulation(params, seed=seed + VALIDATION_SEED_OFFSET)
            revalidated.append(fresh["loss"] if fresh["stable"] else float("nan"))

        clean = [v for v in revalidated if v == v]
        med_reported = statistics.median(reported)
        med_revalidated = statistics.median(clean)
        rows.append({
            "arm": label,
            "reported_best_loss": round(med_reported, 4),
            "revalidated_loss": round(med_revalidated, 4),
            "optimism": round(med_revalidated - med_reported, 4),
            "n_seeds": len(clean),
        })
        print(f"  {label:30} reported {med_reported:.4f} -> revalidated {med_revalidated:.4f} "
              f"(optimism {med_revalidated - med_reported:+.4f})")

    path = OUT / "argon_validation.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {path}")
    print(f"\n(argon truth: epsilon {ARGON_EPSILON:.4f}, sigma {ARGON_SIGMA})")


if __name__ == "__main__":
    main()

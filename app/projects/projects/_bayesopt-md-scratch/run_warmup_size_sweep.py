"""How many warm-up simulations are actually worth buying?

The convergence plot shows the LLM arm flat across runs 1-10: none of runs 2-10
beat run 1. That raises the obvious question — why not take one guess and hand
straight over to the surrogate, saving nine simulations?

Sweeps the warm-up size for both sources, so the comparison has a control. If a
small warm-up works equally well with uniform-random points, the spread is what
matters and the model is not doing the work.

Note the LLM arm here *truncates* a design that was requested as a set of ten.
Asking for a k-point design is not the same thing, and is measured separately.

    cd /Users/federico/Documents/personal/code/agentic-optimiser
    ./.venv/bin/python <this file>
"""

import csv
import json
import sys
from pathlib import Path

REPO = Path("/Users/federico/Documents/personal/code/agentic-optimiser")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))

from generate_data import OUT, run_condition  # noqa: E402

WARMUP_SIZES = [1, 2, 3, 5, 10]


def main() -> None:
    tuned = json.loads((OUT / "winners.json").read_text())["final"]
    rows = []

    for warm_start in ("random", "llm"):
        print(f"\n--- warm_start={warm_start} ---")
        for k in WARMUP_SIZES:
            cond = {**tuned, "warm_start": warm_start, "n_initial_random": k}
            metrics, _ = run_condition(cond)
            rows.append({
                "warm_start": warm_start,
                "n_initial_random": k,
                **{m: round(v, 4) for m, v in metrics.items()},
            })
            print(f"  n_initial={k:<3} loss={metrics['median_best_loss']:.4f} "
                  f"reach={metrics['p_seeds_reach_threshold']:.2f} "
                  f"unstable={metrics['p_unstable']:.3f} "
                  f"steps_to_threshold={metrics['median_steps_to_loss_threshold']:.1f}")

    path = OUT / "warmup_size_sweep.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

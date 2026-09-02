"""Measure the LLM-designed warm-up against uniform-random, at the tuned config.

Run after generate_data.py, with the agentic-optimiser virtualenv:

    cd /Users/federico/Documents/personal/code/agentic-optimiser
    ./.venv/bin/python <this file>

Appends two rows (phase 6_warm_start) to matrix_results.csv and writes the
per-step trajectories for both arms to warmstart_trajectories.csv.

Everything differs in exactly one variable: where the GP's first
n_initial_random points come from. The LLM designs are read from the checked-in
cache, so this is reproducible offline and the API is never called here.
"""

import csv
import json
import sys
from pathlib import Path

REPO = Path("/Users/federico/Documents/personal/code/agentic-optimiser")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))

from generate_data import N_SEEDS, OUT, run_condition  # noqa: E402


def main() -> None:
    winners = json.loads((OUT / "winners.json").read_text())
    tuned = winners["final"]

    rows, arms = [], {}
    for label, warm_start in (("random", "random"), ("llm", "llm")):
        cond = {**tuned, "warm_start": warm_start}
        metrics, results = run_condition(cond)
        arms[label] = results
        rows.append({
            "phase": "6_warm_start",
            "variable": "warm_start",
            "label": f"warm_start={warm_start}",
            "agent": cond["agent"],
            "n_steps": cond["n_steps"],
            "n_initial_random": cond["n_initial_random"],
            "acq_func": cond["acq_func"],
            "unstable_penalty": cond["unstable_penalty"],
            "free_rejection": cond["free_rejection"],
            **{k: round(v, 4) for k, v in metrics.items()},
        })
        print(f"  warm_start={warm_start:7} loss={metrics['median_best_loss']:.4f} "
              f"reach={metrics['p_seeds_reach_threshold']:.2f} "
              f"unstable={metrics['p_unstable']:.3f} "
              f"steps_to_threshold={metrics['median_steps_to_loss_threshold']:.1f}")

    matrix = OUT / "matrix_results.csv"
    existing = list(csv.DictReader(matrix.open()))
    existing = [r for r in existing if r["phase"] != "6_warm_start"]
    fieldnames = list(existing[0].keys())
    with matrix.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing)
        writer.writerows(rows)
    print(f"wrote {matrix} (+2 rows)")

    # Per-step trajectories, for the convergence panel.
    traj = []
    for arm, results in arms.items():
        for seed, (_, step_rows) in enumerate(results):
            best = None
            for row in step_rows:
                loss = float(row["loss"]) if row["loss"] not in ("", None) else None
                if loss is not None and (best is None or loss < best):
                    best = loss
                traj.append({
                    "arm": arm,
                    "seed": seed,
                    "step": int(row["step"]),
                    "loss": loss if loss is not None else "",
                    "best_so_far": best if best is not None else "",
                    "cumulative_cost": float(row["cumulative_cost"]),
                    "stable": row["stable"],
                })
    path = OUT / "warmstart_trajectories.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(traj[0].keys()))
        writer.writeheader()
        writer.writerows(traj)
    print(f"wrote {path} ({len(traj)} rows, {N_SEEDS} seeds per arm)")


if __name__ == "__main__":
    main()

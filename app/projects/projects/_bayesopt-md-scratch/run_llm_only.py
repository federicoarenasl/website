"""The control arm: can a language model optimise this without a surrogate?

Runs LlmAgent over the first N_SEEDS seeds, then re-derives the other three arms
restricted to those same seeds, so a 10-seed LLM run is compared against 10-seed
GP numbers rather than the 20-seed ones reported elsewhere.

Needs credentials (it calls the API ~350 times, roughly $15):

    cd /Users/federico/Documents/personal/code/agentic-optimiser
    set -a && . ./.env && set +a
    ./.venv/bin/python <this file>

Writes llm_only_comparison.csv and llm_only_trajectories.csv; per-run
transcripts land in runs/llm_only/ inside the optimiser repo.
"""

import csv
import json
import statistics
import sys
from pathlib import Path

REPO = Path("/Users/federico/Documents/personal/code/agentic-optimiser")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))

from evals.metrics import DEFAULT_THRESHOLD, aggregate  # noqa: E402
from evals.runner import _make_agent_from_config, run_seeds  # noqa: E402
from optimiser.agent import AGENTS_REGISTRY, AgentConfig  # noqa: E402

from generate_data import OUT  # noqa: E402

N_SEEDS = 10
RUNS_DIR = REPO / "runs" / "llm_only"


def run_llm_arm(tuned: dict) -> tuple[dict, list]:
    """Run the LLM-only agent over the seed set, keeping every transcript."""
    cfg = AgentConfig(
        n_steps=tuned["n_steps"],
        max_budget_usd=float("inf"),
        llm_model="claude-opus-5",
        llm_effort="high",
        verbose=False,
    )
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    results = run_seeds(_make_agent_from_config(AGENTS_REGISTRY["llm"], cfg),
                        N_SEEDS, runs_dir=str(RUNS_DIR))
    return aggregate(results, threshold=DEFAULT_THRESHOLD), results


def _metrics_from_trajectories(path: Path, arm: str) -> dict:
    """Recompute the headline metrics for one arm, on this seed subset only.

    Plain csv rather than pandas: this runs in the optimiser's virtualenv, which
    deliberately carries only what the optimiser itself needs.
    """
    rows = [r for r in csv.DictReader(path.open())
            if r["arm"] == arm and int(r["seed"]) < N_SEEDS]
    best_per_seed, reached, n_steps, n_unstable = {}, set(), 0, 0
    for r in rows:
        seed, n_steps = int(r["seed"]), n_steps + 1
        if r["loss"] in ("", None):
            n_unstable += 1
            continue
        loss = float(r["loss"])
        if loss <= DEFAULT_THRESHOLD:
            reached.add(seed)
        if seed not in best_per_seed or loss < best_per_seed[seed]:
            best_per_seed[seed] = loss
    return {
        "median_best_loss": round(statistics.median(best_per_seed.values()), 4),
        "p_seeds_reach_threshold": round(len(reached) / N_SEEDS, 4),
        "p_unstable": round(n_unstable / max(n_steps, 1), 4),
    }


def main() -> None:
    tuned = json.loads((OUT / "winners.json").read_text())["final"]

    print(f"Running LLM-only over {N_SEEDS} seeds x {tuned['n_steps']} simulations...")
    llm_metrics, llm_results = run_llm_arm(tuned)
    print(f"  loss={llm_metrics['median_best_loss']:.4f} "
          f"reach={llm_metrics['p_seeds_reach_threshold']:.2f} "
          f"unstable={llm_metrics['p_unstable']:.3f}")

    # The other arms, restricted to the same seeds.
    rows = [{"arm": "LLM only (no surrogate)", **{k: round(v, 4) for k, v in llm_metrics.items()}}]
    warm = OUT / "warmstart_trajectories.csv"
    base = OUT / "trajectories.csv"
    for label, path, arm in (
        ("Random search", base, "Random"),
        ("GP (untuned)", base, "GP (untuned)"),
        ("GP (tuned), random warm-up", warm, "random"),
        ("GP (tuned), LLM warm-up", warm, "llm"),
    ):
        rows.append({"arm": label, **_metrics_from_trajectories(path, arm)})

    print(f"\n--- all arms, seeds 0-{N_SEEDS - 1} ---")
    for r in rows:
        print(f"  {r['arm']:30} loss={r['median_best_loss']:.4f} "
              f"reach={r['p_seeds_reach_threshold']:.2f} unstable={r['p_unstable']:.3f}")

    path = OUT / "llm_only_comparison.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {path}")

    # Per-step trajectories for the LLM arm, matching the other CSVs' shape.
    traj = []
    for seed, (_, step_rows) in enumerate(llm_results):
        best = None
        for row in step_rows:
            loss = float(row["loss"]) if row["loss"] not in ("", None) else None
            if loss is not None and (best is None or loss < best):
                best = loss
            traj.append({
                "arm": "llm_only", "seed": seed, "step": int(row["step"]),
                "loss": loss if loss is not None else "",
                "best_so_far": best if best is not None else "",
                "cumulative_cost": float(row["cumulative_cost"]),
                "stable": row["stable"], "violations": row["violations"],
            })
    path = OUT / "llm_only_trajectories.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(traj[0].keys()))
        writer.writeheader()
        writer.writerows(traj)
    print(f"wrote {path} ({len(traj)} rows)")


if __name__ == "__main__":
    main()

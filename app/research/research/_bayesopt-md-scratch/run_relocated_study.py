"""Control study: does the LLM still win when the answer is not the textbook value?

Identical simulator equations, identical prompts, identical tuned settings. The
only change is which measured properties the force field is fitted to, which
moves the optimum from near published coarse-grained values (epsilon 3.10,
sigma 0.460) to somewhere the literature does not point (epsilon 5.20,
sigma 0.550). Selected with MD_TARGET_PRESET=relocated.

The warm-start prompt never states the numeric targets, so the same cached
design is valid under both presets — which makes this an exact control rather
than a re-tuning.

    cd /Users/federico/Documents/personal/code/agentic-optimiser
    set -a && . ./.env && set +a
    MD_TARGET_PRESET=relocated ./.venv/bin/python <this file>
"""

import csv
import json
import sys
from pathlib import Path

REPO = Path("/Users/federico/Documents/personal/code/agentic-optimiser")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))

from evals.metrics import DEFAULT_THRESHOLD, aggregate  # noqa: E402
from evals.runner import _make_agent_from_config, run_seeds  # noqa: E402
from optimiser.agent import AGENTS_REGISTRY, AgentConfig  # noqa: E402
from optimiser.simulator import DENSITY_TARGET, TARGET_PRESET  # noqa: E402

from generate_data import OUT  # noqa: E402

N_SEEDS = 10
RUNS_DIR = REPO / "runs" / "relocated"


def run_arm(agent: str, tuned: dict, **overrides) -> tuple[dict, list]:
    """Run one arm over the seed set under the active target preset."""
    cfg = AgentConfig(
        n_steps=tuned["n_steps"],
        n_initial_random=overrides.get("n_initial_random", tuned["n_initial_random"]),
        acq_func=tuned["acq_func"],
        unstable_penalty=overrides.get("unstable_penalty", tuned["unstable_penalty"]),
        free_rejection=overrides.get("free_rejection", tuned["free_rejection"]),
        warm_start=overrides.get("warm_start", "random"),  # `tag` is for the path only
        max_budget_usd=float("inf"),
        verbose=False,
    )
    results = run_seeds(_make_agent_from_config(AGENTS_REGISTRY[agent], cfg),
                        N_SEEDS, runs_dir=str(RUNS_DIR / overrides.get("tag", agent)))
    return aggregate(results, threshold=DEFAULT_THRESHOLD), results


def main() -> None:
    if TARGET_PRESET != "relocated":
        raise SystemExit("Set MD_TARGET_PRESET=relocated before running this.")
    print(f"preset={TARGET_PRESET}  density target={DENSITY_TARGET} kg/m3\n")

    tuned = json.loads((OUT / "winners.json").read_text())["final"]
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    arms = [
        ("Random search", "random", {"tag": "random"}),
        ("GP (untuned)", "gp", {"unstable_penalty": 10.0, "free_rejection": False, "tag": "gp_untuned"}),
        ("GP (tuned), random warm-up", "gp", {"tag": "gp_tuned"}),
        ("GP (tuned), LLM warm-up", "gp", {"warm_start": "llm", "tag": "gp_llm_warm"}),
        ("LLM only (no surrogate)", "llm", {"tag": "llm_only"}),
    ]

    rows, llm_results = [], None
    for label, agent, overrides in arms:
        print(f"running {label}...")
        metrics, results = run_arm(agent, tuned, **overrides)
        if agent == "llm":
            llm_results = results
        rows.append({"arm": label, **{k: round(v, 4) for k, v in metrics.items()}})
        print(f"  loss={metrics['median_best_loss']:.4f} "
              f"reach={metrics['p_seeds_reach_threshold']:.2f} "
              f"unstable={metrics['p_unstable']:.3f}")

    path = OUT / "relocated_comparison.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {path}")

    traj = []
    for seed, (_, step_rows) in enumerate(llm_results):
        best = None
        for row in step_rows:
            loss = float(row["loss"]) if row["loss"] not in ("", None) else None
            if loss is not None and (best is None or loss < best):
                best = loss
            traj.append({
                "arm": "llm_only_relocated", "seed": seed, "step": int(row["step"]),
                "loss": loss if loss is not None else "",
                "best_so_far": best if best is not None else "",
                "epsilon": row["epsilon"], "sigma": row["sigma"],
                "timestep_fs": row["timestep_fs"],
                "stable": row["stable"], "violations": row["violations"],
            })
    path = OUT / "relocated_llm_trajectories.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(traj[0].keys()))
        writer.writeheader()
        writer.writerows(traj)
    print(f"wrote {path} ({len(traj)} rows)")


if __name__ == "__main__":
    main()

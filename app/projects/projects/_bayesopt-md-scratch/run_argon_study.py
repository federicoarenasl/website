"""The same five arms, against a real molecular-dynamics engine.

Replaces the analytic surrogate with velocity-Verlet Lennard-Jones argon
(optimiser/md.py). Blow-ups, energy drift and measurement noise are emergent
properties of integrating the equations of motion rather than formulas.

The task is parameter recovery: targets were produced by running this engine at
argon's accepted constants (epsilon = 119.8 K * k_B = 0.9961 kJ/mol,
sigma = 0.3405 nm), so success means getting those numbers back from measured
properties alone, and the ground truth is exact.

    cd /Users/federico/Documents/personal/code/agentic-optimiser
    set -a && . ./.env && set +a
    MD_ENGINE=argon ./.venv/bin/python <this file>
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
from optimiser.md import ARGON_EPSILON, ARGON_SIGMA  # noqa: E402
from optimiser.simulator import PARAM_BOUNDS  # noqa: E402

from generate_data import OUT  # noqa: E402

N_SEEDS = 10
N_STEPS = 25  # real MD costs ~11 s a run, so the budget is smaller than the surrogate's
RUNS_DIR = REPO / "runs" / "argon"


def _recovery_error(step_rows: list[list[dict]]) -> dict:
    """How close did the best run of each seed get to argon's true constants?"""
    eps_err, sigma_err = [], []
    for rows in step_rows:
        stable = [r for r in rows if r["loss"] not in ("", None)]
        if not stable:
            continue
        best = min(stable, key=lambda r: float(r["loss"]))
        eps_err.append(abs(float(best["epsilon"]) - ARGON_EPSILON) / ARGON_EPSILON)
        sigma_err.append(abs(float(best["sigma"]) - ARGON_SIGMA) / ARGON_SIGMA)
    return {
        "epsilon_error_pct": round(100 * statistics.median(eps_err), 2) if eps_err else float("nan"),
        "sigma_error_pct": round(100 * statistics.median(sigma_err), 2) if sigma_err else float("nan"),
    }


def run_arm(agent: str, tag: str, **overrides) -> tuple[dict, list]:
    cfg = AgentConfig(
        n_steps=N_STEPS,
        n_initial_random=overrides.get("n_initial_random", 10),
        acq_func="gp_hedge",
        unstable_penalty=overrides.get("unstable_penalty", 0.7),
        free_rejection=overrides.get("free_rejection", True),
        warm_start=overrides.get("warm_start", "random"),
        max_budget_usd=float("inf"),
        verbose=False,
    )
    results = run_seeds(_make_agent_from_config(AGENTS_REGISTRY[agent], cfg),
                        N_SEEDS, runs_dir=str(RUNS_DIR / tag))
    return aggregate(results, threshold=DEFAULT_THRESHOLD), results


def main() -> None:
    print(f"engine=argon  epsilon_true={ARGON_EPSILON:.4f}  sigma_true={ARGON_SIGMA}")
    print(f"bounds: {dict(PARAM_BOUNDS)}")
    print(f"{N_SEEDS} seeds x {N_STEPS} simulations per arm\n")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    arms = [
        ("Random search", "random", "random", {}),
        ("GP (untuned)", "gp", "gp_untuned", {"unstable_penalty": 10.0, "free_rejection": False}),
        ("GP (tuned), random warm-up", "gp", "gp_tuned", {}),
        ("GP (tuned), LLM warm-up", "gp", "gp_llm_warm", {"warm_start": "llm"}),
        ("LLM only (no surrogate)", "llm", "llm_only", {}),
    ]

    rows = []
    for label, agent, tag, overrides in arms:
        print(f"running {label}...", flush=True)
        metrics, results = run_arm(agent, tag, **overrides)
        recovery = _recovery_error([r[1] for r in results])
        rows.append({"arm": label,
                     **{k: round(v, 4) for k, v in metrics.items()},
                     **recovery})
        print(f"  loss={metrics['median_best_loss']:.4f} "
              f"reach={metrics['p_seeds_reach_threshold']:.2f} "
              f"crashed={metrics['p_unstable']:.3f} "
              f"| epsilon off by {recovery['epsilon_error_pct']}%, "
              f"sigma off by {recovery['sigma_error_pct']}%", flush=True)

    path = OUT / "argon_comparison.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

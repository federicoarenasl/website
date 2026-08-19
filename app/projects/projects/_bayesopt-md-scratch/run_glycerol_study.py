"""The five-arm comparison again, on coarse-grained glycerol.

Same arms and same aggregation as run_argon_study.py, so the two tables can be
read side by side. What differs is the task: nine parameters instead of five,
and a target that was generated from a hidden reference rather than published,
so no arm can recall the answer.

    cd /Users/federico/Documents/personal/code/agentic-optimiser
    MD_ENGINE=glycerol ./.venv/bin/python <this file>
"""
import csv
import json
import os
import statistics
import sys
from pathlib import Path

REPO = Path("/Users/federico/Documents/personal/code/agentic-optimiser")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))

if os.environ.get("MD_ENGINE") != "glycerol":
    raise SystemExit("run with MD_ENGINE=glycerol")

from evals.metrics import DEFAULT_THRESHOLD, aggregate  # noqa: E402
from evals.runner import _make_agent_from_config, run_seeds  # noqa: E402
from optimiser.agent import AGENTS_REGISTRY, AgentConfig  # noqa: E402
from optimiser.cg_simulator import REFERENCE_PARAMS  # noqa: E402
from optimiser.simulator import PARAM_BOUNDS  # noqa: E402

from generate_data import OUT  # noqa: E402

# Overridable because the LLM arm is priced per call: measured at $0.2217 a
# call and 30 calls a seed, one seed of that arm costs ~$6.65, so its seed
# count is a budget decision rather than a statistical one. The four CPU arms
# ran at 10 seeds; any arm run at fewer is not comparable on variance.
N_SEEDS = int(os.environ.get("GLYCEROL_N_SEEDS", "10"))
N_STEPS = 30  # nine dimensions needs more than argon's 25; a run costs ~20 s
RUNS_DIR = REPO / "runs" / "glycerol"
# Each arm's aggregate is written the moment it finishes, and the comparison
# table is assembled from those. Arms can then be run in any order or in
# separate processes -- the three that need no LLM can start while the
# warm-start designs are still being generated -- and a crash in one arm does
# not discard the arms that already completed.
PARTIALS = OUT / "_glycerol_partials"

# The four integrator settings are free parameters too, but they are not the
# answer -- these six are what a force field actually consists of.
RECOVERED = ["epsilon", "sigma", "bond_length_nm", "bond_k", "angle_deg", "angle_k"]


def _recovery_error(step_rows: list[list[dict]]) -> dict:
    """Median relative error of each recovered parameter, over seeds."""
    errors: dict[str, list[float]] = {k: [] for k in RECOVERED}
    for rows in step_rows:
        stable = [r for r in rows if r["loss"] not in ("", None)]
        if not stable:
            continue
        best = min(stable, key=lambda r: float(r["loss"]))
        for k in RECOVERED:
            truth = REFERENCE_PARAMS[k]
            errors[k].append(abs(float(best[k]) - truth) / abs(truth))
    out = {f"{k}_error_pct": (round(100 * statistics.median(v), 2) if v else float("nan"))
           for k, v in errors.items()}
    allv = [x for v in errors.values() for x in v]
    out["mean_param_error_pct"] = round(100 * statistics.mean(allv), 2) if allv else float("nan")
    return out


def run_arm(agent: str, tag: str, **overrides):
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


ARMS = [
    ("Random search", "random", "random", {}),
    ("GP (untuned)", "gp", "gp_untuned", {"unstable_penalty": 10.0, "free_rejection": False}),
    ("GP (tuned), random warm-up", "gp", "gp_tuned", {}),
    ("GP (tuned), LLM warm-up", "gp", "gp_llm_warm", {"warm_start": "llm"}),
    ("LLM only (no surrogate)", "llm", "llm_only", {}),
    # Ablation: identical arm with the numeric targets withheld from the
    # prompt (run it with CG_HIDE_TARGETS=1). Separates the LLM reasoning
    # about the physics from the LLM having been handed the target vector,
    # which the surrogate never sees.
    ("LLM only, targets withheld", "llm", "llm_blind", {}),
    # A Bayesian-optimisation baseline configured the way a practitioner
    # would: one surrogate per measured observable rather than one over the
    # scalar loss, the known targets used exactly, and the analytically
    # knowable constraints (minimum image, bond-vibration timestep limit)
    # screened before a simulation is spent. This is the strong BO opponent.
    ("BO, multi-output + constraints", "gpmulti", "gp_multi", {}),
]


def main() -> None:
    print("engine=glycerol  (synthetic target; reference parameters withheld)")
    print(f"{N_SEEDS} seeds x {N_STEPS} simulations per arm")
    print(f"bounds: {dict(PARAM_BOUNDS)}\n")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    arms = list(ARMS)
    PARTIALS.mkdir(parents=True, exist_ok=True)
    wanted = set(sys.argv[1:])
    if wanted:
        arms = [a for a in arms if a[2] in wanted]
        print(f"running only: {', '.join(a[2] for a in arms)}\n")

    for label, agent, tag, overrides in arms:
        cached = PARTIALS / f"{tag}.json"
        if cached.exists():
            print(f"skipping {label} (already done)", flush=True)
            continue
        print(f"running {label}...", flush=True)
        metrics, results = run_arm(agent, tag, **overrides)
        recovery = _recovery_error([r[1] for r in results])
        row = {"arm": label, **{k: round(v, 4) for k, v in metrics.items()}, **recovery}
        cached.write_text(json.dumps(row, indent=2))
        print(f"  loss={metrics['median_best_loss']:.4f} "
              f"reach={metrics['p_seeds_reach_threshold']:.2f} "
              f"crashed={metrics['p_unstable']:.3f} "
              f"| mean parameter error {recovery['mean_param_error_pct']}%", flush=True)

    # Assemble whatever arms exist, in the canonical order.
    order = [a[2] for a in ARMS]
    rows = [json.loads((PARTIALS / f"{t}.json").read_text())
            for t in order if (PARTIALS / f"{t}.json").exists()]
    if not rows:
        return
    path = OUT / "glycerol_comparison.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {path}  ({len(rows)}/{len(order)} arms)")


if __name__ == "__main__":
    main()

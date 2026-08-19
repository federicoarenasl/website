"""How many simulations is a prior worth?

The glycerol study stopped at 30 simulations and concluded that a surrogate has
nothing to learn from at that budget -- three of seven per-observable GPs scored
a negative R^2 on 20 training points. It then made a prediction and did not test
it: that Bayesian optimisation should become competitive somewhere around 60-100
simulations, where the informative observables reach R^2 0.7-0.84.

This runs the same arms out to 150 simulations to find out. If a BO curve crosses
the LLM's 30-simulation result at step N, then the prior was worth N - 30
simulations, and that difference has a price in GPU-hours.

Why one long run gives the whole curve
--------------------------------------
Neither ``BayesOptPolicy`` nor ``MultiGPPolicy`` is budget-aware: nothing in
either reads ``n_steps``, and ``gp_hedge`` does not anneal. A run of 150 steps at
seed s therefore passes through exactly the states a run of 30 steps at seed s
would, so the prefix-minimum of the long run *is* the short run's result. Budgets
are read off one trajectory rather than bought separately -- 150 simulations an
arm instead of 5 x 150.

That is also a testable claim, and ``analyse_budget_curve.py`` tests it: the
step-30 prefix of these runs must reproduce ``glycerol_comparison.csv`` exactly.
If it does not, the assumption is wrong and the curves mean nothing.

Cost: no API calls at all. The LLM warm-start arm reads the designs already
cached in ``llm_warmstart_glycerol.json``; the LLM-only arm is not re-run and
enters the analysis as a fixed reference at 30 simulations.

    cd /Users/federico/Documents/personal/code/agentic-optimiser
    MD_ENGINE=glycerol OMP_NUM_THREADS=1 ./.venv/bin/python \
        /Users/federico/Documents/personal/code/website/app/projects/projects/_bayesopt-md-scratch/run_budget_curve.py
"""
import csv
import json
import os
import sys
import time
from pathlib import Path

REPO = Path("/Users/federico/Documents/personal/code/agentic-optimiser")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))

if os.environ.get("MD_ENGINE") != "glycerol":
    raise SystemExit("run with MD_ENGINE=glycerol")

from evals.runner import _make_agent_from_config, run_seeds  # noqa: E402
from optimiser.agent import AGENTS_REGISTRY, AgentConfig  # noqa: E402

from generate_data import OUT  # noqa: E402

# Overridable only so the plumbing can be smoke-tested on a 2x6 run before
# several hours of CPU are committed. The study itself runs at the defaults.
N_SEEDS = int(os.environ.get("CURVE_N_SEEDS", "12"))  # 12 = one per core; seeds 0-9 are the published study's set
N_STEPS = int(os.environ.get("CURVE_N_STEPS", "150"))  # 5x that study, past the top of its 60-100 prediction
RUNS_DIR = REPO / "runs" / "glycerol_curve"
PARTIALS = OUT / Path(os.environ.get("CURVE_OUT", "_glycerol_curve_partials"))

# The six parameters a force field actually consists of; the other three are
# integrator settings, which are not the answer.
RECOVERED = ["epsilon", "sigma", "bond_length_nm", "bond_k", "angle_deg", "angle_k"]

# Arm definitions are copied verbatim from run_glycerol_study.py rather than
# imported, because that module runs its whole study on import. They must stay
# identical: the step-30 validation below is only meaningful if these are the
# same arms.
ARMS = [
    # The strongest surrogate in the study, and the one the R^2 argument
    # predicts should benefit most from a larger budget. Run first.
    ("BO, multi-output + constraints", "gpmulti", "gp_multi", {}),
    # The prospective bounds experiment: one LLM call per seed elicits a search
    # region up front, and the GP then searches only inside it. Identical in
    # every other respect to "GP (tuned), random warm-up" below, which is its
    # control -- the only difference between the two arms is the box.
    ("BO in LLM-elicited bounds", "gp", "gp_llm_box", {"search_box": "llm"}),
    # Same elicited box, hard crash penalty. The box arm inherited the soft
    # penalty (0.7) tuned on the analytic surrogate, and at 150 simulations that
    # setting is actively harmful: crash rate *rises* with budget on every arm
    # carrying it (25% -> 48% here, 45% -> 79% for LLM warm-up) while the hard
    # penalty falls 33% -> 4.7%. Changing only the penalty isolates that.
    ("BO in LLM-elicited bounds, hard penalty", "gp", "gp_llm_box_hard",
     {"search_box": "llm", "unstable_penalty": 10.0}),
    ("GP (untuned)", "gp", "gp_untuned", {"unstable_penalty": 10.0, "free_rejection": False}),
    ("GP (tuned), LLM warm-up", "gp", "gp_llm_warm", {"warm_start": "llm"}),
    # The worst arm of the study (75% crashed). Included because the crash
    # penalty was the hyperparameter that reversed sign between simulators, and
    # a larger budget is the condition under which that reversal was explained.
    ("GP (tuned), random warm-up", "gp", "gp_tuned", {}),
    ("Random search", "random", "random", {}),
]


def run_arm(agent: str, tag: str, **overrides):
    cfg = AgentConfig(
        n_steps=N_STEPS,
        n_initial_random=overrides.get("n_initial_random", 10),
        acq_func="gp_hedge",
        unstable_penalty=overrides.get("unstable_penalty", 0.7),
        free_rejection=overrides.get("free_rejection", True),
        warm_start=overrides.get("warm_start", "random"),
        search_box=overrides.get("search_box", "declared"),
        max_budget_usd=float("inf"),
        # The stall guard exists to stop a converged run wasting budget. Here a
        # converged run is the measurement -- stopping early would truncate the
        # curve at exactly the budget the experiment is about. Disabled by
        # setting a window no run can fill; repeated proposals are counted in
        # the analysis instead of being acted on.
        stall_window=N_STEPS + 1,
        verbose=False,
    )
    return run_seeds(_make_agent_from_config(AGENTS_REGISTRY[agent], cfg),
                     N_SEEDS, runs_dir=str(RUNS_DIR / tag))


def _rows_for(tag: str, label: str, results) -> list[dict]:
    """Flatten per-seed step logs into long format, one row per simulation."""
    out = []
    for seed, (_summary, steps) in enumerate(results):
        for row in steps:
            rec = {"arm": label, "tag": tag, "seed": seed, "step": int(row["step"]),
                   "stable": row["stable"], "violations": row["violations"],
                   "loss": row["loss"], "cost_usd": row["cost_usd"]}
            rec.update({k: row[k] for k in RECOVERED})
            out.append(rec)
    return out


FIELDS = ["arm", "tag", "seed", "step", "stable", "violations", "loss", "cost_usd", *RECOVERED]


def main() -> None:
    print(f"engine=glycerol  {N_SEEDS} seeds x {N_STEPS} simulations per arm")
    print(f"~23 s a simulation, {N_SEEDS} seeds in parallel "
          f"-> roughly {N_STEPS * 23 / 3600:.1f} h an arm\n", flush=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    PARTIALS.mkdir(parents=True, exist_ok=True)

    wanted = set(sys.argv[1:])
    arms = [a for a in ARMS if not wanted or a[2] in wanted]

    for label, agent, tag, overrides in arms:
        cached = PARTIALS / f"{tag}.csv"
        if cached.exists():
            print(f"skipping {label} (already done)", flush=True)
            continue
        # An arm that reads a cached LLM artifact must not take the queue down
        # with it if that artifact is short of seeds -- this runs unattended.
        if overrides.get("search_box") == "llm":
            from optimiser.bounds import load_cache as load_boxes
            try:
                have = set(load_boxes())
            except FileNotFoundError as exc:
                print(f"  SKIPPING {label}: {exc}", flush=True)
                continue
            missing = set(range(N_SEEDS)) - have
            if missing:
                print(f"  SKIPPING {label}: bounds cache is missing seeds "
                      f"{sorted(missing)}; run `python -m optimiser.bounds` first",
                      flush=True)
                continue
        print(f"running {label}...", flush=True)
        t0 = time.time()
        results = run_arm(agent, tag, **overrides)
        rows = _rows_for(tag, label, results)
        # Written per arm so a crash in a later arm costs only that arm.
        with cached.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        best = [s.best_loss for s, _ in results if s.best_loss is not None]
        print(f"  {len(results)} seeds, {len(rows)} simulations, "
              f"{(time.time() - t0) / 3600:.2f} h, "
              f"best loss {min(best):.4f}-{max(best):.4f}", flush=True)

    # Assemble whatever arms exist so far, in canonical order.
    order = [a[2] for a in ARMS]
    paths = [PARTIALS / f"{t}.csv" for t in order]
    have = [p for p in paths if p.exists()]
    if not have:
        return
    out = OUT / "budget_curve_steps.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for p in have:
            writer.writerows(csv.DictReader(p.open()))
    print(f"\nwrote {out}  ({len(have)}/{len(order)} arms)")
    (OUT / "budget_curve_meta.json").write_text(json.dumps(
        {"n_seeds": N_SEEDS, "n_steps": N_STEPS,
         "arms": {t: lbl for lbl, _, t, _ in ARMS}}, indent=2))


if __name__ == "__main__":
    main()

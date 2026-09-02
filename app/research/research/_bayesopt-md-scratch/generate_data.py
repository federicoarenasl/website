"""Run the staged eval matrix for the Bayesian-optimisation-for-MD post.

Each phase locks in the previous phase's winner, so every row differs from the
current best config in exactly one variable. Winners are picked in code (lowest
median_best_loss, ties broken by reach rate) rather than by eye.

Run with the agentic-optimiser virtualenv (it needs scikit-optimize):

    cd /Users/federico/Documents/personal/code/agentic-optimiser
    ./.venv/bin/python <this file>

Writes into this directory:
  matrix_results.csv     one row per condition, all eval metrics
  winners.json           the config promoted at the end of each phase
  trajectories.csv       per-step best-so-far for GP and Random, all seeds
  landscape.csv          dense loss grid over (timestep, epsilon) at the optimum
  stability_boundary.csv the nominal blow-up cliff, for drawing as a line
  gp_trajectory.csv      one GP run's proposals, for overlaying on the landscape
"""

import csv
import json
import sys
import tempfile
from pathlib import Path

REPO = Path("/Users/federico/Documents/personal/code/agentic-optimiser")
sys.path.insert(0, str(REPO))

from evals.metrics import DEFAULT_THRESHOLD, aggregate  # noqa: E402
from evals.runner import _make_agent_from_config, run_seeds  # noqa: E402
from optimiser.agent import AGENTS_REGISTRY, AgentConfig  # noqa: E402
from optimiser.simulator import (  # noqa: E402
    PARAM_BOUNDS,
    PARAM_NAMES,
    check_violations,
    loss_for,
    max_stable_timestep_fs,
    predict_observables,
)

OUT = Path(__file__).parent
N_SEEDS = 20
EVAL_STEPS = 35
SHARED_SEED = 0  # the run every figure and animation shows  # simulation budget used while tuning the optimiser itself

# Optimum located by the calibration sweep; the landscape slice passes through it.
OPTIMUM = {
    "epsilon": 3.1006,
    "sigma": 0.4601,
    "timestep_fs": 19.1569,
    "thermostat_tau_ps": 1.0,
    "cutoff_nm": 1.0995,
}

# Starting point: a GP set up the obvious way, before any tuning.
BASELINE = {
    "agent": "gp",
    "n_steps": EVAL_STEPS,
    "n_initial_random": 10,
    "acq_func": "gp_hedge",
    "unstable_penalty": 10.0,
    "free_rejection": False,
}


def run_condition(cond: dict) -> tuple[dict, list]:
    """Run one condition over the fixed seed set and return (metrics, results)."""
    cfg = AgentConfig(
        n_steps=cond["n_steps"],
        n_initial_random=cond["n_initial_random"],
        acq_func=cond["acq_func"],
        unstable_penalty=cond["unstable_penalty"],
        free_rejection=cond["free_rejection"],
        warm_start=cond.get("warm_start", "random"),
        max_budget_usd=float("inf"),
        verbose=False,
    )
    results = run_seeds(_make_agent_from_config(AGENTS_REGISTRY[cond["agent"]], cfg), N_SEEDS)
    return aggregate(results, threshold=DEFAULT_THRESHOLD), results


# Seed-to-seed noise on median_best_loss across 20 seeds is a few thousandths,
# so differences below this are not real. Among conditions that tie on loss,
# the one that wastes the fewest simulations on crashes wins.
LOSS_TIE_TOLERANCE = 0.003


def phase(name: str, variable: str, values: list, current: dict, rows: list) -> tuple[dict, dict]:
    """Sweep one variable around the current config; return (winner, its results)."""
    print(f"\n--- {name} (varying {variable}) ---")
    scored = []
    for value in values:
        cond = {**current, variable: value}
        metrics, results = run_condition(cond)
        rows.append({
            "phase": name,
            "variable": variable,
            "label": f"{variable}={value}",
            **{k: cond[k] for k in BASELINE},
            **{k: round(v, 4) for k, v in metrics.items()},
        })
        print(f"  {variable}={str(value):10} loss={metrics['median_best_loss']:.4f} "
              f"reach={metrics['p_seeds_reach_threshold']:.2f} "
              f"unstable={metrics['p_unstable']:.3f} "
              f"thermo_waste={metrics['p_thermostat_violations']:.3f}")
        scored.append((metrics["median_best_loss"], metrics["p_unstable"], value, results))

    best_loss = min(s[0] for s in scored)
    tied = [s for s in scored if s[0] <= best_loss + LOSS_TIE_TOLERANCE]
    tied.sort(key=lambda s: s[1])  # fewest wasted simulations wins the tie
    winner_value, winner_results = tied[0][2], tied[0][3]
    if len(tied) > 1:
        print(f"  ({len(tied)} conditions tie on loss within {LOSS_TIE_TOLERANCE}; "
              f"broken on p_unstable)")
    print(f"  → winner: {variable}={winner_value}")
    return {**current, variable: winner_value}, winner_results


def write_landscape(n_dt: int = 220, n_eps: int = 220) -> None:
    """Dense (timestep, epsilon) slice through the optimum, with crash flags."""
    dt_lo, dt_hi = PARAM_BOUNDS["timestep_fs"]
    eps_lo, eps_hi = PARAM_BOUNDS["epsilon"]
    rows = []
    for i in range(n_eps):
        eps = eps_lo + i * (eps_hi - eps_lo) / (n_eps - 1)
        for j in range(n_dt):
            dt = dt_lo + j * (dt_hi - dt_lo) / (n_dt - 1)
            p = {**OPTIMUM, "epsilon": eps, "timestep_fs": dt}
            obs = predict_observables(p)
            # Nominal boundary: no stochastic jitter, and the thermostat rule is
            # excluded because the policy screens that one before submitting.
            crashed = [v for v in check_violations(p, obs, _NoJitter()) if v != "thermostat"]
            rows.append({
                "epsilon": round(eps, 5),
                "timestep_fs": round(dt, 4),
                "loss": round(loss_for(p), 6) if not crashed else "",
                "violations": "|".join(crashed),
            })
    with (OUT / "landscape.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT / 'landscape.csv'} ({len(rows)} rows)")

    with (OUT / "stability_boundary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epsilon", "max_stable_timestep_fs"])
        writer.writeheader()
        for i in range(n_eps):
            eps = eps_lo + i * (eps_hi - eps_lo) / (n_eps - 1)
            writer.writerow({
                "epsilon": round(eps, 5),
                "max_stable_timestep_fs": round(max_stable_timestep_fs(eps, OPTIMUM["sigma"]), 4),
            })


class _NoJitter:
    """Stand-in RNG that removes the stochastic margin from the crash boundary."""

    def gauss(self, mu, sigma):
        return 0.0

    def uniform(self, a, b):
        return (a + b) / 2

    def random(self):
        return 0.5


def write_trajectories(arms: dict[str, list]) -> None:
    """Dump per-step best-so-far and spend for each arm."""
    rows = []
    for arm, results in arms.items():
        for seed, (_, step_rows) in enumerate(results):
            best = None
            for row in step_rows:
                loss = float(row["loss"]) if row["loss"] not in ("", None) else None
                if loss is not None and (best is None or loss < best):
                    best = loss
                rows.append({
                    "arm": arm,
                    "seed": seed,
                    "step": int(row["step"]),
                    "loss": loss if loss is not None else "",
                    "best_so_far": best if best is not None else "",
                    "cumulative_cost": float(row["cumulative_cost"]),
                    "stable": row["stable"],
                    "violations": row["violations"],
                })
    with (OUT / "trajectories.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT / 'trajectories.csv'} ({len(rows)} rows)")


def median_seed(config: dict, n_seeds: int = N_SEEDS) -> int:
    """Seed whose best loss is closest to the median, for an honest example run.

    Picking a seed by hand risks illustrating the argument with an outlier: seed
    4, chosen arbitrarily at first, turned out to be the 1-in-20 run that
    converges on a too-small timestep and stalls there.
    """
    scores = []
    for seed in range(n_seeds):
        cfg = AgentConfig(
            n_steps=config["n_steps"],
            n_initial_random=config["n_initial_random"],
            acq_func=config["acq_func"],
            unstable_penalty=config["unstable_penalty"],
            free_rejection=config["free_rejection"],
            max_budget_usd=float("inf"),
            seed=seed,
            verbose=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            cfg.runs_dir = tmp
            summary = AGENTS_REGISTRY["gp"](cfg).run()
        scores.append((seed, summary.best_loss))
    losses = sorted(s[1] for s in scores)
    target = losses[len(losses) // 2 - 1]
    chosen = min(scores, key=lambda s: abs(s[1] - target))
    print(f"median seed = {chosen[0]} (best loss {chosen[1]:.4f}, median {target:.4f})")
    return chosen[0]


def write_run_trajectory(config: dict, agent: str, filename: str, seed: int = 4) -> None:
    """One run's proposals in order, for overlaying on the landscape."""
    cfg = AgentConfig(
        n_steps=config["n_steps"],
        n_initial_random=config["n_initial_random"],
        acq_func=config["acq_func"],
        unstable_penalty=config["unstable_penalty"],
        free_rejection=config["free_rejection"],
        warm_start=config.get("warm_start", "random"),
        max_budget_usd=float("inf"),
        seed=seed,
        verbose=False,
    )
    with tempfile.TemporaryDirectory() as tmp:
        cfg.runs_dir = tmp
        AGENTS_REGISTRY[agent](cfg).run()
        rows = list(csv.DictReader(next(Path(tmp).glob("*/steps.csv")).open()))
    keep = ["step"] + PARAM_NAMES + ["stable", "violations", "loss", "best_so_far"]
    with (OUT / filename).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keep)
        writer.writeheader()
        writer.writerows([{k: r[k] for k in keep} for r in rows])
    print(f"wrote {OUT / filename} (agent={agent}, seed={seed})")


def main() -> None:
    rows: list[dict] = []
    winners: dict[str, dict] = {}
    current = dict(BASELINE)

    current, _ = phase("1_penalty", "unstable_penalty",
                       [0.4, 0.5, 0.7, 1.0, 1.3, 2.0, 5.0, 10.0], current, rows)
    winners["1_penalty"] = dict(current)

    current, _ = phase("2_acq", "acq_func", ["gp_hedge", "EI", "PI", "LCB"], current, rows)
    winners["2_acq"] = dict(current)

    current, _ = phase("3_warmup", "n_initial_random", [5, 10, 15, 20], current, rows)
    winners["3_warmup"] = dict(current)

    current, gp_results = phase("4_free_rejection", "free_rejection", [False, True], current, rows)
    winners["4_free_rejection"] = dict(current)

    # Budget sweep is reported, not promoted: the post argues the choice on
    # cost, not on loss alone, so the tuning budget stays at EVAL_STEPS.
    _, _ = phase("5_steps", "n_steps", [15, 25, 35, 50, 70], current, rows)

    final_gp = dict(current)
    winners["final"] = final_gp
    random_cond = {**final_gp, "agent": "random"}
    random_metrics, random_results = run_condition(random_cond)
    rows.append({
        "phase": "6_final", "variable": "agent", "label": "agent=random",
        **{k: random_cond[k] for k in BASELINE},
        **{k: round(v, 4) for k, v in random_metrics.items()},
    })
    baseline_metrics, baseline_results = run_condition(BASELINE)
    rows.append({
        "phase": "6_final", "variable": "agent", "label": "agent=gp_untuned",
        **{k: BASELINE[k] for k in BASELINE},
        **{k: round(v, 4) for k, v in baseline_metrics.items()},
    })
    # The production config: same tuned optimiser, at the budget the cost curve
    # argues for rather than the one used while tuning.
    production = {**final_gp, "n_steps": 50}
    winners["production"] = production
    production_metrics, _ = run_condition(production)
    rows.append({
        "phase": "6_final", "variable": "agent", "label": "agent=gp_tuned_50",
        **{k: production[k] for k in BASELINE},
        **{k: round(v, 4) for k, v in production_metrics.items()},
    })

    print(f"\n--- 6_final ---")
    print(f"  random         loss={random_metrics['median_best_loss']:.4f} reach={random_metrics['p_seeds_reach_threshold']:.2f}")
    print(f"  gp_untuned     loss={baseline_metrics['median_best_loss']:.4f} reach={baseline_metrics['p_seeds_reach_threshold']:.2f}")
    print(f"  gp_tuned@50    loss={production_metrics['median_best_loss']:.4f} reach={production_metrics['p_seeds_reach_threshold']:.2f}")

    with (OUT / "matrix_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {OUT / 'matrix_results.csv'}")
    (OUT / "winners.json").write_text(json.dumps(winners, indent=2))
    print(f"wrote {OUT / 'winners.json'}: {final_gp}")

    write_trajectories({
        "GP (tuned)": gp_results,
        "GP (untuned)": baseline_results,
        "Random": random_results,
    })
    write_landscape()
    # SHARED_SEED, not the median seed: figure 1, trajectory.gif and the blue arm
    # of trajectory-warmstart.gif all show this one run, so it is chosen once, by
    # the paired rule (the seed whose gap between warm-up sources is closest to
    # the median gap). It ranks 6th of 20 on its own — better than typical, but
    # not an outlier.
    write_run_trajectory(final_gp, "gp", "gp_trajectory.csv", seed=SHARED_SEED)
    write_run_trajectory({**final_gp, "warm_start": "llm"}, "gp",
                         "llm_trajectory.csv", seed=SHARED_SEED)


if __name__ == "__main__":
    main()

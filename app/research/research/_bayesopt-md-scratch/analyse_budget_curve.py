"""Read the exchange rate off the budget curves.

Answers one question: at what budget, if any, does a Bayesian optimiser reach
what the LLM reached in 30 simulations? The gap between those two budgets is
what the prior was worth, and multiplying it by the cost of a simulation prices
it.

Everything here is a prefix computation over ``budget_curve_steps.csv``, which
is valid only because neither policy is budget-aware -- see the header of
``run_budget_curve.py``. The first thing this script does is check that
assumption rather than assert it.

    ./.venv/bin/python analyse_budget_curve.py
"""
import csv
import json
import math
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/Users/federico/Documents/personal/code/agentic-optimiser")

from glycerol_runs import steps_csv  # noqa: E402
from optimiser.cg_simulator import REFERENCE_PARAMS  # noqa: E402

HERE = Path(__file__).parent
STEPS = HERE / "budget_curve_steps.csv"
PUBLISHED = HERE / "glycerol_comparison.csv"

NOISE_FLOOR = 0.0351
THRESHOLD = 0.12
RECOVERED = ["epsilon", "sigma", "bond_length_nm", "bond_k", "angle_deg", "angle_k"]
# The 30-simulation LLM-only result this experiment is measured against. Three
# seeds, so it is a soft target -- the crossover budget inherits that softness.
LLM_REF_LOSS = 0.1247
LLM_SEEDS = [0, 1, 2]
N_BOOT = 4000


def load_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


# Arm directory -> label, so a run still in progress can be read straight from
# the agent's own output. The finished CSV is authoritative; this exists so the
# analysis can be exercised end to end while the study is hours from done,
# rather than finding a bug in it at the end.
LIVE_ARMS = {
    "gp_multi": "BO, multi-output + constraints",
    "gp_llm_box": "BO in LLM-elicited bounds",
    "gp_llm_box_hard": "BO in LLM-elicited bounds, hard penalty",
    "gp_untuned": "GP (untuned)",
    "gp_llm_warm": "GP (tuned), LLM warm-up",
    "gp_tuned": "GP (tuned), random warm-up",
    "random": "Random search",
}
LIVE_DIR = Path("/Users/federico/Documents/personal/code/agentic-optimiser/runs/glycerol_curve")


def load_live() -> list[dict]:
    """Assemble the long table from run directories mid-flight.

    Seeds are still running, so they are ragged: seed 3 may be 40 simulations
    ahead of seed 7. Every median here is therefore taken over whichever seeds
    have reached that budget, which is fine for checking the pipeline and wrong
    for a headline -- the finished CSV is what gets reported.
    """
    rows = []
    for tag, label in LIVE_ARMS.items():
        for seed_dir in sorted((LIVE_DIR / tag).glob("run_*")):
            steps = sorted(seed_dir.glob("*/steps.csv"), key=lambda p: p.stat().st_size)
            if not steps:
                continue
            seed = int(seed_dir.name.split("_")[1])
            for r in load_rows(steps[-1]):
                rows.append({"arm": label, "tag": tag, "seed": seed, **r})
    return rows


def by_arm_seed(rows: list[dict]) -> dict[str, dict[int, list[dict]]]:
    out: dict[str, dict[int, list[dict]]] = {}
    for r in rows:
        out.setdefault(r["arm"], {}).setdefault(int(r["seed"]), []).append(r)
    for seeds in out.values():
        for steps in seeds.values():
            steps.sort(key=lambda r: int(r["step"]))
    return out


def best_so_far(steps: list[dict]) -> list[float]:
    """Running minimum loss, with ``inf`` until the first run survives."""
    out, best = [], math.inf
    for r in steps:
        if r["loss"] not in ("", None):
            best = min(best, float(r["loss"]))
        out.append(best)
    return out


def best_row_within(steps: list[dict], budget: int) -> dict | None:
    stable = [r for r in steps[:budget] if r["loss"] not in ("", None)]
    return min(stable, key=lambda r: float(r["loss"])) if stable else None


def param_error(row: dict) -> float:
    """Mean relative error over the six force-field parameters, in percent."""
    return 100 * statistics.mean(
        abs(float(row[k]) - REFERENCE_PARAMS[k]) / abs(REFERENCE_PARAMS[k])
        for k in RECOVERED)


def median_ci(values: list[float], rng: random.Random) -> tuple[float, float, float]:
    """Median with a bootstrap 95% interval over seeds.

    The study this follows reported medians without intervals and flagged that
    as its main statistical weakness, so every headline number here carries one.
    """
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return math.inf, math.inf, math.inf
    med = statistics.median(values) if len(finite) == len(values) else math.inf
    boots = []
    for _ in range(N_BOOT):
        sample = [values[rng.randrange(len(values))] for _ in values]
        fin = [v for v in sample if math.isfinite(v)]
        boots.append(statistics.median(sample) if len(fin) == len(sample) else math.inf)
    boots = [b for b in boots if math.isfinite(b)]
    if not boots:
        return med, math.inf, math.inf
    boots.sort()
    return med, boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]


def llm_curves() -> dict[int, list[float]]:
    """Per-step best-so-far for the three LLM-only seeds that completed."""
    out = {}
    for seed in LLM_SEEDS:
        path = steps_csv("llm_only", seed)
        if path is None:
            continue
        out[seed] = best_so_far(load_rows(path))
    return out


def validate_against_published(data, published: Path) -> list[str]:
    """The step-30 prefix on seeds 0-9 must reproduce the published study.

    Same arms, same seeds, same simulator, and policies that cannot see the
    budget: the long runs must pass through the short runs' results. If they do
    not, prefix-reading is invalid and every curve below is meaningless.
    """
    if not published.exists():
        return ["published comparison not found -- validation skipped"]
    pub = {r["arm"]: r for r in load_rows(published)}
    notes = []
    for arm, seeds in sorted(data.items()):
        if arm not in pub:
            continue
        # Only seeds that have actually reached 30 simulations can be compared
        # against a 30-simulation study. Mid-flight this is usually nobody, and
        # reporting a short prefix as a FAIL would cry wolf.
        common = [s for s in seeds if s < 10 and len(seeds[s]) >= 30]
        if len(common) < 10:
            notes.append(f"--   {arm}: only {len(common)}/10 published seeds have "
                         f"reached 30 simulations -- validation deferred")
            continue
        losses = []
        for s in sorted(common):
            row = best_row_within(seeds[s], 30)
            losses.append(float(row["loss"]) if row else math.inf)
        got = statistics.median(losses)
        want = float(pub[arm]["median_best_loss"])
        ok = math.isfinite(got) and abs(got - want) < 5e-4
        notes.append(f"{'OK  ' if ok else 'FAIL'} {arm}: step-30 median "
                     f"{got:.4f} vs published {want:.4f}")
    return notes


def main() -> None:
    live = "--live" in sys.argv
    if live:
        rows = load_live()
        if not rows:
            raise SystemExit("no run directories yet")
        counts = {}
        for r in rows:
            counts.setdefault(r["arm"], set()).add(int(r["seed"]))
        print("LIVE READ -- seeds are ragged, medians are provisional")
        for arm, seeds in counts.items():
            steps = [int(r["step"]) for r in rows if r["arm"] == arm]
            print(f"  {arm}: {len(seeds)} seeds, {min(steps)}-{max(steps)} steps")
        print()
    elif not STEPS.exists():
        raise SystemExit(f"no data yet at {STEPS} (use --live to read in-progress runs)")
    else:
        rows = load_rows(STEPS)
    data = by_arm_seed(rows)
    rng = random.Random(0)
    n_steps = max(int(r["step"]) for r in rows)

    print("=" * 78)
    print("VALIDATION -- prefix-reading assumption")
    print("=" * 78)
    for note in validate_against_published(data, PUBLISHED):
        print("  " + note)

    llm = llm_curves()
    llm_final = statistics.median([c[-1] for c in llm.values()]) if llm else LLM_REF_LOSS

    print()
    print("=" * 78)
    print(f"BUDGET CURVES -- median best-so-far, {n_steps} simulations")
    print("=" * 78)
    marks = [10, 20, 30, 50, 75, 100, 125, 150]
    marks = [m for m in marks if m <= n_steps]
    header = "arm".ljust(34) + "".join(f"{m:>9}" for m in marks)
    print(header)
    print("-" * len(header))
    curves = {}
    for arm, seeds in data.items():
        curve = {s: best_so_far(st) for s, st in seeds.items()}
        curves[arm] = curve
        cells = []
        for m in marks:
            vals = [c[m - 1] for c in curve.values() if len(c) >= m]
            med, _lo, _hi = median_ci(vals, rng)
            cells.append("    --   " if not math.isfinite(med) else f"{med:>9.4f}")
        print(arm.ljust(34) + "".join(cells))
    if llm:
        cells = []
        for m in marks:
            vals = [c[m - 1] for c in llm.values() if len(c) >= m]
            cells.append(f"{statistics.median(vals):>9.4f}" if vals else "    --   ")
        print(("LLM only (3 seeds, 30 max)").ljust(34) + "".join(cells))

    print()
    print("=" * 78)
    print("THE EXCHANGE RATE")
    print("=" * 78)
    print(f"  LLM-only reached {llm_final:.4f} in 30 simulations "
          f"(noise floor {NOISE_FLOOR}, threshold {THRESHOLD})")
    summary = {"n_steps": n_steps, "llm_ref": llm_final, "arms": {}}
    for arm, curve in curves.items():
        n_seeds = len(curve)
        cross_llm = cross_thr = None
        for step in range(1, n_steps + 1):
            vals = [c[step - 1] for c in curve.values() if len(c) >= step]
            if len(vals) < n_seeds:
                break
            med = statistics.median(vals)
            if cross_llm is None and med <= llm_final:
                cross_llm = step
            if cross_thr is None and med <= THRESHOLD:
                cross_thr = step
        final_vals = [c[-1] for c in curve.values()]
        med, lo, hi = median_ci(final_vals, rng)
        # Per-seed crossing: how many seeds ever get there, regardless of what
        # the median does. A median that never crosses can still hide seeds
        # that do.
        seeds_cross = sum(1 for c in curve.values() if min(c) <= llm_final)
        best_row = [best_row_within(data[arm][s], n_steps) for s in sorted(data[arm])]
        perr = [param_error(r) for r in best_row if r]
        print(f"\n  {arm}")
        print(f"    final median loss   {med:.4f}  [95% CI {lo:.4f}, {hi:.4f}]")
        print(f"    crosses LLM's {llm_final:.4f}  "
              f"{'at simulation ' + str(cross_llm) if cross_llm else 'never (' + str(n_steps) + ' simulations)'}")
        print(f"    crosses {THRESHOLD}         "
              f"{'at simulation ' + str(cross_thr) if cross_thr else 'never'}")
        print(f"    seeds ever reaching it   {seeds_cross}/{n_seeds}")
        print(f"    mean parameter error at {n_steps}: {statistics.median(perr):.2f}%"
              if perr else "    mean parameter error: --")
        summary["arms"][arm] = {
            "n_seeds": n_seeds,
            "final_median_loss": None if not math.isfinite(med) else round(med, 4),
            "ci95": [None if not math.isfinite(lo) else round(lo, 4),
                     None if not math.isfinite(hi) else round(hi, 4)],
            "crossover_vs_llm": cross_llm,
            "crossover_vs_threshold": cross_thr,
            "seeds_ever_reaching_llm": f"{seeds_cross}/{n_seeds}",
            "median_param_error_pct": round(statistics.median(perr), 2) if perr else None,
        }

    # Does a bigger budget buy accuracy, or only a lower loss? The study's
    # sharpest finding was that the strongest surrogate found its lowest losses
    # further from the truth, by exploiting compensating directions between
    # parameters. More budget could fix that or deepen it.
    print()
    print("=" * 78)
    print("LOSS vs PARAMETER RECOVERY -- median over seeds")
    print("=" * 78)
    header = "arm".ljust(34) + "".join(f"{m:>9}" for m in marks)
    print(header + "   (mean parameter error %)")
    print("-" * len(header))
    for arm, seeds in data.items():
        cells = []
        for m in marks:
            errs = [param_error(r) for r in
                    (best_row_within(st, m) for st in seeds.values()) if r]
            cells.append(f"{statistics.median(errs):>9.1f}" if errs else "    --   ")
        print(arm.ljust(34) + "".join(cells))
        summary["arms"][arm]["param_error_by_budget"] = {
            str(m): (round(statistics.median([param_error(r) for r in
                     (best_row_within(st, m) for st in seeds.values()) if r]), 2)
                     if any(best_row_within(st, m) for st in seeds.values()) else None)
            for m in marks}

    # Crash rate by phase. If the surrogate ever learns the feasible region --
    # the thing the LLM knew for free -- it shows up here as a falling rate.
    print()
    print("=" * 78)
    print("CRASH RATE BY PHASE -- does the surrogate learn the boundary?")
    print("=" * 78)
    bands = [(1, 30), (31, 60), (61, 100), (101, n_steps)]
    bands = [(a, b) for a, b in bands if a <= n_steps]
    header = "arm".ljust(34) + "".join(f"{f'{a}-{b}':>12}" for a, b in bands)
    print(header)
    print("-" * len(header))
    for arm, seeds in data.items():
        cells = []
        for a, b in bands:
            flags = [r["stable"] for st in seeds.values() for r in st
                     if a <= int(r["step"]) <= b]
            rate = sum(1 for f in flags if f in ("False", "false", "0")) / len(flags) if flags else None
            cells.append(f"{rate:>11.1%} " if rate is not None else "        --  ")
        print(arm.ljust(34) + "".join(cells))
        summary["arms"][arm]["crash_rate_by_phase"] = {
            f"{a}-{b}": round(sum(1 for r in (x for st in seeds.values() for x in st)
                                  if a <= int(r["step"]) <= b
                                  and r["stable"] in ("False", "false", "0"))
                              / max(1, len([1 for st in seeds.values() for r in st
                                            if a <= int(r["step"]) <= b])), 4)
            for a, b in bands}

    (HERE / "budget_curve_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {HERE / 'budget_curve_summary.json'}")


if __name__ == "__main__":
    main()

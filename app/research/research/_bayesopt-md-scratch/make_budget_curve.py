"""The exchange rate, drawn: how much budget a prior is worth.

Two panels, one argument.

Left -- median best-so-far against simulations spent, with the LLM's
30-simulation result carried across as a horizontal rule. The quantity being
read is *horizontal*: how far right a surrogate has to travel to reach a height
the prior reached at 30. That distance is the result, so it is annotated as a
distance rather than left for the reader to measure.

Right -- mean parameter error against the same budget. Loss and recovery are
not the same axis, and the study's sharpest finding was that the strongest
surrogate found its lowest losses furthest from the truth. If that holds at
150 simulations, these two panels disagree with each other, and that
disagreement is the honest headline.

    python3 app/research/research/_bayesopt-md-scratch/make_budget_curve.py
"""
import math
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SCRATCH = Path(__file__).parent
sys.path.insert(0, str(SCRATCH))
sys.path.insert(0, "/Users/federico/Documents/personal/code/agentic-optimiser")

from glycerol_runs import steps_csv  # noqa: E402
from analyse_budget_curve import (  # noqa: E402
    NOISE_FLOOR, RECOVERED, THRESHOLD, best_so_far, best_row_within,
    llm_curves, load_rows, param_error,
)
from make_animation import (  # noqa: E402
    AQUA, BASE, BLUE, CRITICAL, GRID, INK, INK_MUTED, INK_SECONDARY, ORANGE,
    SURFACE, VIOLET, YELLOW, round_axes, style_ax,
)

OUT = Path("/Users/federico/Documents/personal/code/website/public/bayesopt-for-md-simulators")
DPI = 200

# Colours follow the arm identities already established in the animations, so
# the two figures can be read against each other.
ARMS = [
    ("Random search", YELLOW),
    ("GP (untuned)", BLUE),
    ("GP (tuned), random warm-up", YELLOW),
    ("BO, multi-output + constraints", INK_SECONDARY),
    ("GP (tuned), LLM warm-up", VIOLET),
    ("BO in LLM-elicited bounds", AQUA),
    ("BO in LLM-elicited bounds, hard penalty", ORANGE),
]
LLM_COLOUR = ORANGE


def median_curve(curves: dict[int, list[float]], n: int) -> np.ndarray:
    """Median across seeds at each budget; NaN where a majority has no result."""
    out = np.full(n, np.nan)
    for i in range(n):
        vals = [c[i] for c in curves.values() if len(c) > i]
        if not vals:
            continue
        med = statistics.median(vals)
        out[i] = med if math.isfinite(med) else np.nan
    return out


def band(curves: dict[int, list[float]], n: int, lo=25, hi=75):
    los, his = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        vals = [c[i] for c in curves.values() if len(c) > i and math.isfinite(c[i])]
        if len(vals) < 3:
            continue
        los[i], his[i] = np.percentile(vals, lo), np.percentile(vals, hi)
    return los, his


def main() -> None:
    rows = load_rows(SCRATCH / "budget_curve_steps.csv")
    df = pd.DataFrame(rows)
    df["step"] = df["step"].astype(int)
    df["seed"] = df["seed"].astype(int)
    n = int(df["step"].max())
    x = np.arange(1, n + 1)

    curves, errors = {}, {}
    for arm, _ in ARMS:
        sub = df[df["arm"] == arm]
        if sub.empty:
            continue
        per_seed = {s: g.sort_values("step").to_dict("records")
                    for s, g in sub.groupby("seed")}
        curves[arm] = {s: best_so_far(st) for s, st in per_seed.items()}
        marks = list(range(5, n + 1, 5))
        errors[arm] = (marks, [statistics.median(
            [param_error(r) for r in (best_row_within(st, m) for st in per_seed.values()) if r]
            or [np.nan]) for m in marks])

    llm = llm_curves()
    llm_med = median_curve(llm, max(len(c) for c in llm.values())) if llm else None
    llm_final = float(llm_med[-1]) if llm_med is not None else 0.1247

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.6), facecolor=SURFACE)
    for a in (ax, ax2):
        a.set_facecolor(SURFACE)

    # ---- left: loss against budget -------------------------------------
    ax.axhspan(0, NOISE_FLOOR, color=BASE, alpha=0.28, lw=0)
    ax.axhline(THRESHOLD, color=INK_MUTED, lw=0.9, ls=(0, (4, 3)), zorder=2)
    ax.text(n * 0.035, THRESHOLD * 1.08, "good enough", ha="left", va="bottom",
            fontsize=8, color=INK_MUTED)
    ax.text(n * 0.035, NOISE_FLOOR * 1.10, "noise floor", ha="left", va="bottom",
            fontsize=8, color=INK_MUTED)

    for arm, colour in ARMS:
        if arm not in curves:
            continue
        med = median_curve(curves[arm], n)
        lo, hi = band(curves[arm], n)
        ax.fill_between(x, lo, hi, color=colour, alpha=0.10, lw=0)
        ax.plot(x, med, color=colour, lw=1.9, label=arm, zorder=4)

    if llm_med is not None:
        m = len(llm_med)
        ax.plot(np.arange(1, m + 1), llm_med, color=LLM_COLOUR, lw=2.2,
                label="LLM only (3 seeds)", zorder=5)
        ax.plot([m], [llm_med[-1]], "o", ms=5, color=LLM_COLOUR, zorder=6)
        # The LLM stops at 30; carrying its final value across the panel is what
        # makes the horizontal reading possible, so it is drawn as a rule rather
        # than left implicit.
        ax.plot([m, n], [llm_final, llm_final], color=LLM_COLOUR, lw=1.0,
                ls=(0, (2, 3)), alpha=0.85, zorder=5)

        # Annotate the crossing: the budget at which the best surrogate arm
        # reaches what the prior reached at 30.
        crossing = None
        for arm, colour in ARMS:
            if arm not in curves:
                continue
            med = median_curve(curves[arm], n)
            hits = np.where(np.nan_to_num(med, nan=np.inf) <= llm_final)[0]
            if len(hits) and (crossing is None or hits[0] + 1 < crossing[0]):
                crossing = (int(hits[0] + 1), arm, colour)
        if crossing:
            step, arm, colour = crossing
            ax.annotate("", xy=(step, llm_final), xytext=(m, llm_final),
                        arrowprops=dict(arrowstyle="<->", color=INK, lw=1.2,
                                        shrinkA=0, shrinkB=0))
            ax.text((m + step) / 2, llm_final * 0.80,
                    f"{step - m} simulations", ha="center", va="top",
                    fontsize=9, color=INK, fontweight="bold")
            ax.plot([step], [llm_final], "o", ms=5, color=colour, zorder=6)
        else:
            ax.text(n * 0.55, llm_final * 0.74,
                    "no surrogate arm reaches it\nin %d simulations" % n,
                    ha="center", va="top", fontsize=9, color=INK, fontweight="bold")

    ax.set_yscale("log")
    ax.set_xlim(1, n)
    ax.set_xlabel("simulations spent", fontsize=9, color=INK_SECONDARY)
    ax.set_ylabel("best loss so far (median over seeds)", fontsize=9, color=INK_SECONDARY)
    ax.set_title("What a budget buys", fontsize=11, color=INK, loc="left", pad=10)
    style_ax(ax, grid_axis="y")
    ax.legend(frameon=False, fontsize=8, loc="upper right", labelcolor=INK_SECONDARY)

    # ---- right: parameter recovery against budget -----------------------
    for arm, colour in ARMS:
        if arm not in errors:
            continue
        marks, vals = errors[arm]
        ax2.plot(marks, vals, color=colour, lw=1.9, label=arm)
    if llm:
        paths = {s: steps_csv("llm_only", s) for s in llm}
        per_seed = {s: load_rows(p) for s, p in paths.items() if p}
        marks = list(range(5, 31, 5))
        vals = [statistics.median([param_error(r) for r in
                (best_row_within(st, m) for st in per_seed.values()) if r] or [np.nan])
                for m in marks]
        ax2.plot(marks, vals, color=LLM_COLOUR, lw=2.2, label="LLM only")

    ax2.set_xlim(1, n)
    ax2.set_xlabel("simulations spent", fontsize=9, color=INK_SECONDARY)
    ax2.set_ylabel("mean error on the six force-field parameters (%)",
                   fontsize=9, color=INK_SECONDARY)
    ax2.set_title("What it recovers", fontsize=11, color=INK, loc="left", pad=10)
    style_ax(ax2, grid_axis="y")

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "glycerol-budget-curve.png"
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

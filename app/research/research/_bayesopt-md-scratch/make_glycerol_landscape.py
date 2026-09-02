"""Every arm's trajectory over the bond-length / angle loss landscape.

The slice holds the other seven parameters at the hidden reference values, so
its minimum sits exactly on the answer -- which is what makes the figure worth
drawing. All four arms converge toward a bond length of about 0.300 nm, while
the true minimum is at 0.3126: the LLM warm-start designs read the mean-bond
observable (0.2996 nm) as if it were the equilibrium bond length, and nothing
downstream ever corrected it.

Honest caveat, stated here because the picture cannot state it: a trajectory
point's own loss is not the value of the field beneath it. Each proposal has
its own seven other parameters, so the field is the landscape the optimiser
would have faced had it got everything else exactly right. The markers show
where each arm looked; the field shows what it was looking for.

    python3 app/research/research/_bayesopt-md-scratch/make_glycerol_landscape.py
"""

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.animation import FuncAnimation  # noqa: E402

SCRATCH = Path(__file__).parent
sys.path.insert(0, str(SCRATCH))
from make_animation import (  # noqa: E402
    AQUA, BASE, BLUE, CRITICAL, INK, INK_MUTED, INK_SECONDARY, LOSS_CMAP,
    ORANGE, SURFACE, VIOLET, YELLOW, _write_animation, round_axes, round_corner_elbow,
    style_ax,
)
from glycerol_runs import all_seeds, steps_csv  # noqa: E402
from make_molecule_animation import LOSS_FLOOR, THRESHOLD, TRUTH, load  # noqa: E402

OUT = Path("/Users/federico/Documents/personal/code/website/public/bayesopt-for-md-simulators")
REPO = Path("/Users/federico/Documents/personal/code/agentic-optimiser")
DPI = 130

ARMS = [
    ("random", "Random search", YELLOW),
    ("gp_untuned", "GP (untuned)", BLUE),
    ("gp_tuned", "GP (tuned), random warm-up", AQUA),
    ("gp_llm_warm", "GP (tuned), LLM warm-up", VIOLET),
    ("llm_only", "LLM only (no surrogate)", ORANGE),
]
# The value every LLM design proposed, by reading the mean-bond observable
# literally. Drawn as a reference line because it is the point of the figure.
NAIVE_BOND = 0.2996


def _best_series(rows):
    out, cur = [], float("inf")
    for r in rows:
        if r["best_so_far"] is not None:
            cur = r["best_so_far"]
        out.append(cur)
    return out


def main() -> None:
    land = pd.read_csv(SCRATCH / "glycerol_landscape.csv")
    land["loss"] = pd.to_numeric(land["loss"], errors="coerce")
    bonds = np.sort(land["bond_length_nm"].unique())
    angles = np.sort(land["angle_deg"].unique())
    grid = land.pivot(index="bond_length_nm", columns="angle_deg", values="loss").values
    crashed = np.isnan(grid)

    data = []
    for tag, label, colour in ARMS:
        src = [steps_csv(tag, 0)] if steps_csv(tag, 0) else []
        if not src:
            continue
        rows = load(src[0])
        data.append({"label": label, "colour": colour, "rows": rows,
                     "best": _best_series(rows)})
    n = max(len(d["rows"]) for d in data)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.5), dpi=DPI,
                                  gridspec_kw={"width_ratios": [1.12, 1.0]})
    fig.patch.set_facecolor(BASE)
    for a in (ax, ax2):
        a.set_facecolor(SURFACE)

    finite = grid[~crashed]
    levels = np.linspace(float(np.nanmin(finite)), float(np.nanpercentile(finite, 92)), 24)
    field = ax.contourf(angles, bonds, np.clip(grid, None, levels[-1]),
                        levels=levels, cmap=LOSS_CMAP, extend="neither", zorder=1)
    # ContourSet.collections was removed in matplotlib 3.10; the set itself is
    # the artist now. Rasterising matters: a vector contour mesh at this
    # density emits tens of thousands of paths.
    field.set_rasterized(True)
    if crashed.any():
        ax.contourf(angles, bonds, crashed.astype(float), levels=[0.5, 1.5],
                    colors=[BASE], alpha=0.85, zorder=2)

    cbar = fig.colorbar(field, ax=ax, pad=0.02)
    cbar.set_label("Loss vs measured properties", fontsize=8.5, color=INK_SECONDARY)
    cbar.ax.tick_params(labelsize=7.5, colors=INK_SECONDARY)
    cbar.outline.set_visible(False)

    ax.set_xlabel("Equilibrium angle θ₀ (degrees)")
    ax.set_ylabel("Equilibrium bond length b₀ (nm)")
    style_ax(ax)

    # The answer, and the wrong answer the observable invites.
    ax.axhline(TRUTH["bond_length_nm"], color=INK, lw=0.9, ls="-", alpha=0.55, zorder=4)
    ax.text(angles[0] + 1.5, TRUTH["bond_length_nm"], "truth  0.3126", ha="left",
            va="bottom", fontsize=7.5, color=INK, zorder=5)
    ax.axhline(NAIVE_BOND, color=CRITICAL, lw=0.9, ls=(0, (4, 3)), alpha=0.75, zorder=4)
    ax.text(angles[0] + 1.5, NAIVE_BOND, "observable read literally  0.2996",
            ha="left", va="top", fontsize=7.5, color=CRITICAL, zorder=5)
    ax.plot([TRUTH["angle_deg"]], [TRUTH["bond_length_nm"]], marker="*",
            ms=13, color=INK, mec=SURFACE, mew=1.0, zorder=6)

    # No connecting lines. These are paths through a 2D projection of a 9D
    # search, so consecutive proposals land arbitrarily far apart and the
    # segments joining them describe nothing -- they only produce spaghetti
    # that hides the point of the figure, which is where each arm's best
    # parameters ended up relative to the truth.
    bests = [ax.plot([], [], marker="D", ms=9, color=d["colour"], mec=SURFACE,
                     mew=1.4, ls="none", zorder=10)[0] for d in data]
    points = [ax.scatter([], [], s=18, color=d["colour"], edgecolors="none",
                         alpha=0.40, zorder=8) for d in data]
    heads = [ax.plot([], [], marker="o", ms=7, color=d["colour"], mec=SURFACE,
                     mew=1.0, alpha=0.85, zorder=9)[0] for d in data]

    ax2.set_xlim(1, n)
    top = max(max(b for b in d["best"] if math.isfinite(b)) for d in data)
    ax2.set_yscale("log")
    ax2.set_ylim(LOSS_FLOOR * 0.7, top * 1.25)
    ax2.set_xlabel("Simulations run")
    ax2.set_ylabel("Best loss so far")
    style_ax(ax2)
    ax2.axhline(THRESHOLD, color=INK_MUTED, lw=1.0, ls="--", zorder=1)
    ax2.text(1.4, THRESHOLD, "good enough", ha="left", va="bottom", fontsize=8,
             color=INK_MUTED)
    ax2.axhline(LOSS_FLOOR, color=INK_MUTED, lw=0.9, ls=":", zorder=1)
    ax2.text(1.4, LOSS_FLOOR, "noise floor", ha="left", va="bottom", fontsize=8,
             color=INK_MUTED)
    curves = [ax2.plot([], [], color=d["colour"], lw=2.2, zorder=3,
                       label=d["label"])[0] for d in data]
    leg = ax2.legend(loc="upper right", frameon=False, fontsize=7.5, ncol=1,
                     handlelength=1.5)
    for t in leg.get_texts():
        t.set_color(INK_SECONDARY)
    round_corner_elbow(ax2)

    fig.tight_layout()
    round_axes(ax, [field, *bests, *points, *heads])
    round_axes(ax2, curves)
    round_axes(cbar.ax, [cbar.solids], radius_in=0.045)

    def frame(k):
        for d, best_m, pt, head, curve in zip(data, bests, points, heads, curves):
            m = min(k, len(d["rows"]) - 1)
            seen = d["rows"][:m + 1]
            xs = [r["angle_deg"] for r in seen]
            ys = [r["bond_length_nm"] for r in seen]
            pt.set_offsets(np.column_stack([xs, ys]) if xs else np.empty((0, 2)))
            head.set_data([xs[-1]], [ys[-1]])
            # Diamond marks the best parameter set found so far -- the only
            # point in the trace that the arm would actually hand over.
            scored = [r for r in seen if r["loss"] is not None]
            if scored:
                b = min(scored, key=lambda r: r["loss"])
                best_m.set_data([b["angle_deg"]], [b["bond_length_nm"]])
            pts = [(x + 1, y) for x, y in enumerate(d["best"][:m + 1]) if math.isfinite(y)]
            curve.set_data([q[0] for q in pts], [q[1] for q in pts])
        return []

    anim = FuncAnimation(fig, frame, frames=n, blit=False)
    _write_animation(anim, OUT / "glycerol-landscape.gif")
    plt.close(fig)


if __name__ == "__main__":
    main()

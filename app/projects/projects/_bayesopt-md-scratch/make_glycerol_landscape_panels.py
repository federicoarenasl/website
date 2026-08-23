"""One loss-landscape panel per arm: where each strategy actually looked.

The single-panel version stacks 150 proposals from five arms onto one field,
which hides the thing worth seeing. How an arm *distributes* its samples is the
difference between the strategies -- random search sprays across the whole
slice, a surrogate concentrates late, and an LLM proposing every point barely
moves. Overplotting turns all three into the same cloud.

Field and caveat are as in make_glycerol_landscape.py: the other seven
parameters are held at the hidden reference values, so the minimum sits on the
answer, and a marker's own loss is not the value of the field beneath it.

    python3 app/projects/projects/_bayesopt-md-scratch/make_glycerol_landscape_panels.py
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
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402

SCRATCH = Path(__file__).parent
sys.path.insert(0, str(SCRATCH))
from glycerol_runs import representative_seed, steps_csv  # noqa: E402
from make_animation import (  # noqa: E402
    AQUA, BASE, BLUE, INK, INK_MUTED, INK_SECONDARY, LOSS_CMAP, ORANGE,
    SURFACE, VIOLET, YELLOW, _write_animation, round_axes, round_corner_elbow,
    style_ax,
)
from make_glycerol_landscape import NAIVE_BOND, _best_series  # noqa: E402
from make_molecule_animation import LOSS_FLOOR, THRESHOLD, TRUTH, load  # noqa: E402

OUT = Path("/Users/federico/Documents/personal/code/website/public/bayesopt-for-md-simulators")
DPI = 130
ARMS = [
    ("random", "Random search", YELLOW),
    ("gp_untuned", "GP (untuned)", BLUE),
    ("gp_tuned", "GP (tuned), random", AQUA),
    ("gp_llm_warm", "GP (tuned), LLM warm-up", VIOLET),
    ("llm_only", "LLM only", ORANGE),
]


def main() -> None:
    land = pd.read_csv(SCRATCH / "glycerol_landscape.csv")
    land["loss"] = pd.to_numeric(land["loss"], errors="coerce")
    bonds = np.sort(land["bond_length_nm"].unique())
    angles = np.sort(land["angle_deg"].unique())
    grid = land.pivot(index="bond_length_nm", columns="angle_deg", values="loss").values
    crashed = np.isnan(grid)
    finite = grid[~crashed]
    levels = np.linspace(float(np.nanmin(finite)), float(np.nanpercentile(finite, 92)), 24)

    data = []
    for tag, label, colour in ARMS:
        src = steps_csv(tag, representative_seed(tag))
        if src is None:
            continue
        rows = load(src)
        data.append({"label": label, "colour": colour, "rows": rows,
                     "best": _best_series(rows)})
    n = max(len(d["rows"]) for d in data)

    fig = plt.figure(figsize=(11.0, 6.0), dpi=DPI)
    fig.patch.set_facecolor(BASE)
    gs = fig.add_gridspec(2, len(data) + 1, height_ratios=[1.3, 1.0],
                          width_ratios=[1] * len(data) + [0.055],
                          hspace=0.26, wspace=0.10,
                          left=0.05, right=0.965, top=0.945, bottom=0.085)
    axes = [fig.add_subplot(gs[0, i]) for i in range(len(data))]
    cax = fig.add_subplot(gs[0, len(data)])
    ax2 = fig.add_subplot(gs[1, :len(data)])
    for a in (*axes, ax2):
        a.set_facecolor(SURFACE)

    scatters, heads, bests, trails = [], [], [], []
    for i, (ax, d) in enumerate(zip(axes, data)):
        field = ax.contourf(angles, bonds, np.clip(grid, None, levels[-1]),
                            levels=levels, cmap=LOSS_CMAP, extend="neither", zorder=1)
        field.set_rasterized(True)
        if crashed.any():
            ax.contourf(angles, bonds, crashed.astype(float), levels=[0.5, 1.5],
                        colors=[BASE], alpha=0.85, zorder=2)
        ax.axhline(TRUTH["bond_length_nm"], color=INK, lw=0.8, alpha=0.5, zorder=4)
        ax.axhline(NAIVE_BOND, color="#d03b3b", lw=0.8, ls=(0, (4, 3)),
                   alpha=0.65, zorder=4)
        ax.plot([TRUTH["angle_deg"]], [TRUTH["bond_length_nm"]], marker="*", ms=11,
                color=INK, mec=SURFACE, mew=0.9, zorder=6)
        # The path in search order. Older segments fade, so the line reads as
        # a direction of travel rather than a tangle -- with 30 proposals in a
        # panel a uniform line just fills it.
        trail = LineCollection([], linewidths=1.0, zorder=6)
        ax.add_collection(trail)
        trails.append(trail)
        scatters.append(ax.scatter([], [], s=16, color=d["colour"], alpha=0.5,
                                   edgecolors="none", zorder=7))
        heads.append(ax.plot([], [], marker="o", ms=6, color=d["colour"],
                             mec=SURFACE, mew=0.9, zorder=8)[0])
        bests.append(ax.plot([], [], marker="D", ms=8, color=d["colour"],
                             mec=SURFACE, mew=1.2, ls="none", zorder=9)[0])
        ax.set_title(d["label"], fontsize=8, color=INK_SECONDARY, pad=5)
        ax.set_xlabel("θ₀ (deg)", fontsize=8)
        if i == 0:
            ax.set_ylabel("b₀ (nm)", fontsize=8.5)
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=7)
        style_ax(ax)
        round_corner_elbow(ax)

    cbar = fig.colorbar(field, cax=cax)
    cbar.set_label("Loss", fontsize=8, color=INK_SECONDARY)
    cbar.ax.tick_params(labelsize=7, colors=INK_SECONDARY)
    cbar.outline.set_visible(False)

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
    curves = [ax2.plot([], [], color=d["colour"], lw=2.0, zorder=3,
                       label=d["label"])[0] for d in data]
    leg = ax2.legend(loc="upper right", frameon=False, fontsize=7.5, ncol=2)
    for t in leg.get_texts():
        t.set_color(INK_SECONDARY)
    round_corner_elbow(ax2)
    for ax, sc, hd, bs, tr in zip(axes, scatters, heads, bests, trails):
        round_axes(ax, [sc, hd, bs, tr])
    round_axes(ax2, curves)
    round_axes(cbar.ax, [cbar.solids], radius_in=0.045)

    def frame(k):
        for d, sc, hd, bs, tr, curve in zip(data, scatters, heads, bests, trails, curves):
            m = min(k, len(d["rows"]) - 1)
            seen = d["rows"][:m + 1]
            xs = [r["angle_deg"] for r in seen]
            ys = [r["bond_length_nm"] for r in seen]
            sc.set_offsets(np.column_stack([xs, ys]) if xs else np.empty((0, 2)))
            hd.set_data([xs[-1]], [ys[-1]])
            pts_xy = np.column_stack([xs, ys])
            if len(pts_xy) > 1:
                segs = np.stack([pts_xy[:-1], pts_xy[1:]], axis=1)
                ramp = np.linspace(0.10, 0.75, len(segs))
                tr.set_segments(segs)
                tr.set_color([to_rgba(d["colour"], a) for a in ramp])
            else:
                tr.set_segments([])
            scored = [r for r in seen if r["loss"] is not None]
            if scored:
                b = min(scored, key=lambda r: r["loss"])
                bs.set_data([b["angle_deg"]], [b["bond_length_nm"]])
            pts = [(x + 1, y) for x, y in enumerate(d["best"][:m + 1]) if math.isfinite(y)]
            curve.set_data([q[0] for q in pts], [q[1] for q in pts])
        return []

    anim = FuncAnimation(fig, frame, frames=n, blit=False)
    _write_animation(anim, OUT / "glycerol-landscape-panels.gif")
    plt.close(fig)


if __name__ == "__main__":
    main()

"""Six recovered parameters in three panels, connected only where they improved.

Joining every consecutive proposal draws the optimiser's exploration, which is
mostly noise: BO deliberately alternates probing and exploiting, so successive
points sit far apart and the line says nothing. Connecting only the proposals
that *improved on the best loss so far* leaves the handful of moves that
actually built the answer -- typically three to six per run -- which is sparse
enough to overlay all five arms in one panel.

The three panels group the six parameters being recovered by what they do:
equilibrium geometry, stiffness, and non-bonded. The three integrator settings
are omitted; they have to be chosen well but they are not the force field.

Faint dots are every proposal, for context. Solid line and rings are the
improvement path. Diamond is where the arm finished. Star is the answer.

    python3 app/research/research/_bayesopt-md-scratch/make_glycerol_improvements.py
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
REPO = Path("/Users/federico/Documents/personal/code/agentic-optimiser")
sys.path.insert(0, str(REPO))

from glycerol_runs import representative_seed, steps_csv  # noqa: E402
from make_animation import (  # noqa: E402
    AQUA, BASE, BLUE, CRITICAL, INK, INK_MUTED, INK_SECONDARY, LOSS_CMAP,
    ORANGE, SURFACE, VIOLET, YELLOW, _write_animation, round_axes,
    round_corner_elbow, style_ax,
)
from make_glycerol_landscape import NAIVE_BOND, _best_series  # noqa: E402
from make_molecule_animation import LOSS_FLOOR, THRESHOLD, TRUTH  # noqa: E402
from optimiser.cg_simulator import PARAM_BOUNDS, REFERENCE_PARAMS  # noqa: E402

OUT = Path("/Users/federico/Documents/personal/code/website/public/bayesopt-for-md-simulators")
DPI = 130

# Three arms, same budget, same information. Both the surrogate and the LLM
# are given the reference targets -- which is the realistic setup, since a
# practitioner fitting a force field has the reference data in hand. The
# surrogate uses them through a multi-output model (one GP per measured
# observable, loss composed analytically) plus the analytically knowable
# feasibility constraints, so it is the strong configuration rather than the
# scalar-loss one.
#
# Random search is the floor, not a competitor.
ARMS = [
    ("random", "Random search", YELLOW),
    ("gp_multi", "Bayesian optimisation", BLUE),
    ("llm_only", "LLM", ORANGE),
]

# Each panel's background field: 400 real simulations with the other seven
# parameters held at the hidden reference values, so every field's minimum sits
# on the answer. Column names differ because the geometry field predates the
# generalised sweep script.
PANELS = [
    ("angle_deg", "bond_length_nm", "Equilibrium geometry", "θ₀ (deg)", "b₀ (nm)",
     ("glycerol_landscape.csv", "angle_deg", "bond_length_nm")),
    ("bond_k", "angle_k", "Stiffness", "k_bond (kJ/mol/nm²)", "k_angle (kJ/mol)",
     ("glycerol_field_bond_k__angle_k.csv", "x", "y")),
    ("sigma", "epsilon", "Non-bonded", "σ (nm)", "ε (kJ/mol)",
     ("glycerol_field_sigma__epsilon.csv", "x", "y")),
]


def load_field(spec):
    """Grid a field CSV into (x values, y values, loss array, crash mask)."""
    fname, xcol, ycol = spec
    df = pd.read_csv(SCRATCH / fname)
    df["loss"] = pd.to_numeric(df["loss"], errors="coerce")
    xs = np.sort(df[xcol].unique())
    ys = np.sort(df[ycol].unique())
    grid = df.pivot(index=ycol, columns=xcol, values="loss").values
    return xs, ys, grid, np.isnan(grid)


def load_rows(path: Path) -> list[dict]:
    import csv
    out = []
    for r in csv.DictReader(path.open()):
        rec = {k: (float(r[k]) if r[k] else None) for k in
               (*PARAM_BOUNDS, "loss", "best_so_far")}
        rec["stable"] = r["stable"] == "True"
        out.append(rec)
    return out


def improvement_indices(rows) -> list[int]:
    """Indices of proposals that improved on the best loss so far."""
    idx, best = [], math.inf
    for i, r in enumerate(rows):
        if r["loss"] is not None and r["loss"] < best - 1e-12:
            best = r["loss"]
            idx.append(i)
    return idx


def main() -> None:
    data = []
    for tag, label, colour in ARMS:
        src = steps_csv(tag, representative_seed(tag))
        if src is None:
            continue
        rows = load_rows(src)
        data.append({"label": label, "colour": colour, "rows": rows,
                     "best": _best_series(rows), "imp": improvement_indices(rows)})
    n = max(len(d["rows"]) for d in data)

    fig = plt.figure(figsize=(11.0, 6.2), dpi=DPI)
    fig.patch.set_facecolor(BASE)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1.0],
                          width_ratios=[1, 1, 1, 0.045],
                          hspace=0.30, wspace=0.22,
                          left=0.062, right=0.935, top=0.94, bottom=0.082)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])
    ax2 = fig.add_subplot(gs[1, :3])
    for a in (*axes, ax2):
        a.set_facecolor(SURFACE)

    fields = [load_field(spec) for *_h, spec in PANELS]
    # One colour scale across all three panels, taken from the pooled losses,
    # so a shade means the same thing in each. Per-panel scaling would make the
    # shallowest basin look as deep as the sharpest.
    pooled = np.concatenate([g[~m].ravel() for _x, _y, g, m in fields])
    levels = np.linspace(float(pooled.min()), float(np.percentile(pooled, 90)), 24)

    artists = {i: {} for i in range(3)}
    for i, (xk, yk, title, xlab, ylab, _spec) in enumerate(PANELS):
        ax = axes[i]
        fx, fy, fgrid, fmask = fields[i]
        cf = ax.contourf(fx, fy, np.clip(fgrid, None, levels[-1]), levels=levels,
                         cmap=LOSS_CMAP, extend="neither", zorder=1)
        cf.set_rasterized(True)
        if fmask.any():
            ax.contourf(fx, fy, fmask.astype(float), levels=[0.5, 1.5],
                        colors=[BASE], alpha=0.85, zorder=2)
        ax.set_xlim(*PARAM_BOUNDS[xk])
        ax.set_ylim(*PARAM_BOUNDS[yk])
        ax.set_title(title, fontsize=9, color=INK_SECONDARY, pad=6)
        ax.set_xlabel(xlab, fontsize=8.5)
        ax.set_ylabel(ylab, fontsize=8.5)
        ax.tick_params(labelsize=7.5)
        style_ax(ax)
        # The answer, plus (geometry panel only) the value a literal reading of
        # the mean-bond observable gives -- the anchor every LLM design used.
        if yk == "bond_length_nm":
            ax.axhline(NAIVE_BOND, color=CRITICAL, lw=0.9, ls=(0, (4, 3)),
                       alpha=0.7, zorder=3)
            ax.text(0.5, 0.03, "dashed: observable read literally",
                    transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=7, color=CRITICAL, zorder=4)
        ax.plot([REFERENCE_PARAMS[xk]], [REFERENCE_PARAMS[yk]], marker="*", ms=15,
                color=INK, mec=SURFACE, mew=1.1, zorder=9)
        for d in data:
            artists[i][d["label"]] = {
                "dots": ax.scatter([], [], s=11, color=d["colour"], alpha=0.22,
                                   edgecolors="none", zorder=5),
                "path": ax.plot([], [], color=d["colour"], lw=1.6, alpha=0.9,
                                marker="o", ms=4.5, mfc=SURFACE, mew=1.3,
                                mec=d["colour"], zorder=7)[0],
                "best": ax.plot([], [], marker="D", ms=8, color=d["colour"],
                                mec=SURFACE, mew=1.2, ls="none", zorder=8)[0],
            }
        # The angle stiffness is the least identifiable of the nine: a 2.5x
        # error lifts the loss only to ~1.7x the noise floor. At one seed per
        # grid point that direction is dominated by sampling noise, which is
        # why this field's minimum sits at k_angle 66 rather than the true
        # 38.6. The vertical structure here is not meaningful.
        if yk == "angle_k":
            ax.text(0.5, 0.965, "k_angle weakly constrained", transform=ax.transAxes,
                    ha="center", va="top", fontsize=7, color=INK_SECONDARY,
                    style="italic", zorder=10)
        round_corner_elbow(ax)

    cbar = fig.colorbar(cf, cax=cax)
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
    leg = ax2.legend(loc="upper right", frameon=False, fontsize=8, ncol=2)
    for t in leg.get_texts():
        t.set_color(INK_SECONDARY)
    round_corner_elbow(ax2)
    for i, ax in enumerate(axes):
        round_axes(ax, [a for g in artists[i].values() for a in g.values()])
    round_axes(ax2, curves)

    def frame(k):
        for i, (xk, yk, *_rest) in enumerate(PANELS):  # noqa: B007
            for d in data:
                g = artists[i][d["label"]]
                m = min(k, len(d["rows"]) - 1)
                seen = d["rows"][:m + 1]
                xs = [r[xk] for r in seen]
                ys = [r[yk] for r in seen]
                g["dots"].set_offsets(np.column_stack([xs, ys]) if xs
                                      else np.empty((0, 2)))
                imp = [j for j in d["imp"] if j <= m]
                g["path"].set_data([d["rows"][j][xk] for j in imp],
                                   [d["rows"][j][yk] for j in imp])
                if imp:
                    g["best"].set_data([d["rows"][imp[-1]][xk]],
                                       [d["rows"][imp[-1]][yk]])
        for d, curve in zip(data, curves):
            m = min(k, len(d["best"]) - 1)
            pts = [(x + 1, y) for x, y in enumerate(d["best"][:m + 1])
                   if math.isfinite(y)]
            curve.set_data([q[0] for q in pts], [q[1] for q in pts])
        return []

    anim = FuncAnimation(fig, frame, frames=n, blit=False)
    _write_animation(anim, OUT / "glycerol-improvements.gif")
    plt.close(fig)
    print("improvement counts (seed 0):")
    for d in data:
        print(f"   {d['label']:<26}{len(d['imp'])} improving moves")


if __name__ == "__main__":
    main()

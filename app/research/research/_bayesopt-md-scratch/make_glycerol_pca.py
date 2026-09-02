"""Two views of the search that the bond-angle slice cannot give.

Left: all nine parameters projected onto their first two principal components.
Every parameter is first mapped to [0,1] by its own bounds, because otherwise
bond_k (400-6000) would dominate the variance and the projection would just be
a plot of bond_k. There is deliberately no background loss field: the
projection is many-to-one, so two points with very different losses can land in
the same place and a field over these axes would be multi-valued. The
projection shows where each arm looked, not what it found.

Right: distance from the true parameter vector in the full nine dimensions,
again after normalising each axis by its bounds. This is the view the
projections cannot give -- nothing is discarded, and it answers the question
the figures are really about: did the arm approach the answer, or just find a
low loss somewhere else?

    python3 app/research/research/_bayesopt-md-scratch/make_glycerol_pca.py
"""

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SCRATCH = Path(__file__).parent
sys.path.insert(0, str(SCRATCH))
REPO = Path("/Users/federico/Documents/personal/code/agentic-optimiser")
sys.path.insert(0, str(REPO))

from glycerol_runs import all_seeds  # noqa: E402
from make_animation import (  # noqa: E402
    AQUA, BASE, BLUE, INK, INK_MUTED, INK_SECONDARY, SURFACE, VIOLET, YELLOW,
    round_axes, round_corner_elbow, style_ax,
)
from optimiser.cg_simulator import PARAM_BOUNDS, PARAM_NAMES, REFERENCE_PARAMS  # noqa: E402

OUT = Path("/Users/federico/Documents/personal/code/website/public/bayesopt-for-md-simulators")
DPI = 200
ARMS = [
    ("random", "Random search", YELLOW),
    ("gp_untuned", "GP (untuned)", BLUE),
    ("gp_tuned", "GP (tuned), random warm-up", AQUA),
    ("gp_llm_warm", "GP (tuned), LLM warm-up", VIOLET),
]


def unit(row) -> np.ndarray:
    """Map a parameter set into the unit cube using its own bounds."""
    return np.array([(float(row[k]) - PARAM_BOUNDS[k][0])
                     / (PARAM_BOUNDS[k][1] - PARAM_BOUNDS[k][0]) for k in PARAM_NAMES])


def main() -> None:
    import csv

    truth = unit(REFERENCE_PARAMS)
    arms = []
    for tag, label, colour in ARMS:
        seeds = []
        for f in all_seeds(tag):
            rows = list(csv.DictReader(f.open()))
            if rows:
                seeds.append(rows)
        if seeds:
            arms.append({"label": label, "colour": colour, "seeds": seeds})

    # Basis fitted on every proposal from every arm and seed, so all arms are
    # drawn in one shared frame rather than each in its own.
    allpts = np.array([unit(r) for a in arms for s in a["seeds"] for r in s])
    centre = allpts.mean(axis=0)
    _, sv, vt = np.linalg.svd(allpts - centre, full_matrices=False)
    basis = vt[:2]
    var = sv**2 / np.sum(sv**2)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.4), dpi=DPI)
    fig.patch.set_facecolor(BASE)
    for a in (ax, ax2):
        a.set_facecolor(SURFACE)

    # ---- left: PCA of the proposals ------------------------------------
    for a in arms:
        pts = np.array([unit(r) for r in a["seeds"][0]])
        proj = (pts - centre) @ basis.T
        ax.plot(proj[:, 0], proj[:, 1], color=a["colour"], lw=0.7, alpha=0.35, zorder=3)
        ax.scatter(proj[:, 0], proj[:, 1], s=22, color=a["colour"], alpha=0.75,
                   edgecolors="none", zorder=4, label=a["label"])
    tproj = (truth - centre) @ basis.T
    ax.plot([tproj[0]], [tproj[1]], marker="*", ms=17, color=INK, mec=SURFACE,
            mew=1.2, zorder=6)
    ax.annotate("truth", tproj, textcoords="offset points", xytext=(9, 6),
                fontsize=8, color=INK, zorder=6)
    ax.set_xlabel(f"PC1 ({var[0]*100:.0f}% of variance)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.0f}%)")
    style_ax(ax)
    round_corner_elbow(ax)

    # ---- right: distance to truth in all nine dimensions ---------------
    for a in arms:
        curves = []
        for s in a["seeds"]:
            d = [float(np.linalg.norm(unit(r) - truth)) for r in s]
            best, cur = [], math.inf
            for x in d:
                cur = min(cur, x)
                best.append(cur)
            curves.append(best)
        n = max(len(c) for c in curves)
        padded = np.array([c + [c[-1]] * (n - len(c)) for c in curves])
        med = np.median(padded, axis=0)
        lo, hi = np.percentile(padded, [25, 75], axis=0)
        x = np.arange(1, n + 1)
        ax2.fill_between(x, lo, hi, color=a["colour"], alpha=0.13, lw=0, zorder=2)
        ax2.plot(x, med, color=a["colour"], lw=2.2, zorder=3, label=a["label"])

    ax2.set_xlabel("Simulations run")
    ax2.set_ylabel("Closest approach to truth (normalised 9-D distance)")
    ax2.set_xlim(1, 30)
    ax2.set_ylim(0, None)
    style_ax(ax2)
    leg = ax2.legend(loc="upper right", frameon=False, fontsize=7.5)
    for t in leg.get_texts():
        t.set_color(INK_SECONDARY)
    round_corner_elbow(ax2)

    fig.tight_layout()
    round_axes(ax, [])
    round_axes(ax2, [])
    fig.savefig(OUT / "glycerol-pca.svg", facecolor=BASE, bbox_inches="tight")
    plt.close(fig)
    kb = (OUT / "glycerol-pca.svg").stat().st_size / 1024
    print(f"wrote glycerol-pca.svg ({kb:.0f} KB)")
    print(f"PC1 {var[0]*100:.1f}%, PC2 {var[1]*100:.1f}%, together {sum(var[:2])*100:.1f}%")
    print("PC1 loadings:")
    for name, w in sorted(zip(PARAM_NAMES, basis[0]), key=lambda t: -abs(t[1])):
        print(f"   {name:<20}{w:+.3f}")


if __name__ == "__main__":
    main()

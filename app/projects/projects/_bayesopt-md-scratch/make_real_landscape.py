"""The real-target loss landscape, and where the search actually went.

Stage 1's landscape was drawn over bonded geometry, because that is what a
nine-parameter fit was searching. Here the bonded parameters are frozen and the
free ones are the bead's identity -- how deep the well is and how big the bead
is -- so the field is loss over (epsilon, sigma), cohesion against packing.

Every background cell is a real NPT simulation scored against experimental
density and enthalpy of vaporisation, so a crash is a hole in the field rather
than a large number, and the minimum is where the model comes closest to real
glycerol rather than to a known answer.

This figure needs no reconstruction: epsilon, sigma and the loss were logged
correctly throughout. Only DHvap was missing, and it is not used here.

Slice caveat, stated on the figure: the cutoff is held at the value the BO
baseline settled on. Seeds that chose a different cutoff are still plotted, and
their loss belongs to their own slice, not this one.

    python3 make_real_landscape.py
"""
import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SCRATCH = Path(__file__).parent
sys.path.insert(0, str(SCRATCH))
from make_animation import (  # noqa: E402
    BASE, CRITICAL, INK, INK_MUTED, INK_SECONDARY, LOSS_CMAP, SURFACE, style_ax,
)

LANDSCAPE = SCRATCH / "real_landscape.csv"
RUNS = Path("/Users/federico/Documents/personal/code/agentic-optimiser/runs")
OUT = Path("/Users/federico/Documents/personal/code/website/public/bayesopt-for-md-simulators")
CUTOFF = 1.1159


def load_field():
    rows = list(csv.DictReader(LANDSCAPE.open()))
    eps = sorted({float(r["epsilon"]) for r in rows})
    sig = sorted({float(r["sigma"]) for r in rows})
    grid = np.full((len(sig), len(eps)), np.nan)
    for r in rows:
        i, j = sig.index(float(r["sigma"])), eps.index(float(r["epsilon"]))
        if r["loss"] != "":
            grid[i, j] = float(r["loss"])
    return np.array(eps), np.array(sig), grid


def seed_bests(run_dir):
    """Each seed's best (epsilon, sigma, loss, cutoff) -- all logged, none inferred."""
    out = []
    for d in sorted(run_dir.glob("run_*")):
        cands = sorted(d.glob("*/steps.csv"), key=lambda p: p.stat().st_size)
        if not cands:
            continue
        rows = [r for r in csv.DictReader(cands[-1].open())
                if r["loss"] not in ("", None)]
        if rows:
            b = min(rows, key=lambda r: float(r["loss"]))
            out.append((float(b["epsilon"]), float(b["sigma"]),
                        float(b["loss"]), float(b["cutoff_nm"])))
    return out


def main():
    eps, sig, grid = load_field()
    fig, ax = plt.subplots(figsize=(7.4, 5.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    # Crashed cells first, so the field is drawn as holes in a hatched ground
    # rather than as an interpolated surface over regions that returned nothing.
    ax.set_facecolor("#f0efe9")
    finite = grid[np.isfinite(grid)]
    levels = np.linspace(float(finite.min()), float(np.percentile(finite, 90)), 20)
    mesh = ax.contourf(eps, sig, grid, levels=levels, cmap=LOSS_CMAP, extend="max")
    cb = fig.colorbar(mesh, ax=ax, pad=0.02)
    cb.set_label("loss against experiment", fontsize=8, color=INK_SECONDARY)
    cb.ax.tick_params(labelsize=7, colors=INK_MUTED)

    ei, si = np.meshgrid(eps, sig)
    crashed = ~np.isfinite(grid)
    ax.plot(ei[crashed], si[crashed], "x", color=INK_MUTED, ms=3.5,
            alpha=0.55, mew=0.8, zorder=3)

    k = np.unravel_index(np.nanargmin(grid), grid.shape)
    ax.plot([eps[k[1]]], [sig[k[0]]], marker="*", ms=17, color=CRITICAL,
            markeredgecolor="white", markeredgewidth=0.8, zorder=7,
            label=f"field minimum {grid[k]:.4f}")

    bests = seed_bests(RUNS / "glycerol_real") or seed_bests(RUNS / "glycerol_real_v1")
    if bests:
        # Marker size carries the cutoff each seed chose: this field is one
        # slice, and a seed that settled far from it is not really on this map.
        for e, s, loss, cut in bests:
            off = abs(cut - CUTOFF)
            ax.plot([e], [s], "o", ms=6, markerfacecolor="none",
                    markeredgecolor=INK, mew=1.0 + 1.6 * (off < 0.05),
                    alpha=0.9, zorder=6)
        ax.plot([], [], "o", markerfacecolor="none", markeredgecolor=INK,
                label=f"best of each BO seed (n={len(bests)})")

    ax.set_xlabel(r"$\epsilon$  — well depth (kJ mol$^{-1}$)", fontsize=9,
                  color=INK_SECONDARY)
    ax.set_ylabel(r"$\sigma$  — bead diameter (nm)", fontsize=9,
                  color=INK_SECONDARY)
    ax.set_title("Cohesion against packing, scored on real glycerol",
                 fontsize=11, color=INK, loc="left", pad=10)
    ax.text(0.99, 1.015, f"slice at cutoff {CUTOFF:.3f} nm  ·  "
            f"{int(np.isfinite(grid).sum())}/{grid.size} ran",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            color=INK_MUTED)
    style_ax(ax)
    # Boxed and on the pale infeasible ground: unboxed grey text sat on the
    # darkest contours and was unreadable.
    leg = ax.legend(fontsize=8, loc="lower right", labelcolor=INK,
                    frameon=True, framealpha=0.9)
    leg.get_frame().set_facecolor(SURFACE)
    leg.get_frame().set_edgecolor(BASE)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "glycerol-real-landscape.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    print(f"wrote {path}")
    print(f"  field minimum {grid[k]:.4f} at epsilon={eps[k[1]]}, sigma={sig[k[0]]}")
    print(f"  {int(np.isfinite(grid).sum())}/{grid.size} cells ran, "
          f"{int(crashed.sum())} crashed")


if __name__ == "__main__":
    main()

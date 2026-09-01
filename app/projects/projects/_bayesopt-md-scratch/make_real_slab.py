"""A slab through each fitted liquid, at true bead size.

The whole-box figure cannot show packing. Projecting 375 beads through 2.5 nm of
depth gives an areal coverage of six to seven times, so every panel is six deep
in overlapping discs and saturates: coverage grows as r^2/L^2 while packing
grows as r^3/L^3, which compresses exactly the signal being looked for. Shrinking
the drawn beads to keep it legible made that worse.

A slab fixes it by removing the depth. One slice, 0.35 nm thick -- about one bead
diameter -- taken through the middle of the box, with every bead drawn at its
true sigma/2. Coverage comes out near 1.0, so discs touch and gaps show, and what
you are looking at is the actual local structure of the liquid rather than a
projection of it.

Two things this makes legible that the box figure cannot:

* **It is a liquid.** Disordered, touching, with holes -- not a lattice and not a
  gas. At kT/epsilon ~ 0.55 a monatomic Lennard-Jones system would freeze at this
  temperature; the bonded connectivity is what keeps this one fluid.
* **Bead size against separation.** sigma spans 0.354 to 0.395 across the arms,
  and at true radius that difference is visible as how much neighbouring discs
  overlap.

And one thing to be honest about: the arms' *packing fractions* differ by 16%
(0.582 to 0.678), but in a fixed-thickness slab that shows up as only a few
percent of areal coverage, because a larger box puts fewer beads in the slab.
The dominant density signal is still the box size, which is why the square and
its dashed reference are drawn here too.

    MD_ENGINE=glycerol_real python3 make_real_slab.py
"""
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation  # noqa: E402

SCRATCH = Path(__file__).parent
sys.path.insert(0, str(SCRATCH))
sys.path.insert(0, "/Users/federico/Documents/personal/code/agentic-optimiser")

from make_animation import (  # noqa: E402
    AQUA, BASE, BLUE, CRITICAL, INK, INK_MUTED, INK_SECONDARY, ORANGE, VIOLET,
    _write_animation,
)
from make_real_box import (ARM_SPECS, RHO_STAR, SEED, SNAPSHOTS,  # noqa: E402
                           reference_box_nm)

OUT = Path("/Users/federico/Documents/personal/code/website/public/bayesopt-for-md-simulators")
WHITE = "#ffffff"
# One bead diameter. Thicker and the discs stack again; thinner and there is not
# enough liquid in the slice to read as one.
SLAB_NM = 0.35
# Three states rather than fifty: continuous churn was distracting, and the
# comparison lives in the final state, which is why it is held far longer.
N_STATES = 3
FRAME_MS, HOLD_MS = 1100, 4000


def slab_indices(pos: np.ndarray, box: float, thickness: float) -> np.ndarray:
    """Beads whose centres lie in a central slice, along the viewing axis."""
    z = pos[:, 2] % box
    return np.abs(z - box / 2.0) <= thickness / 2.0


def draw(ax, pos, box, sigma, colour, ref, label, stamp=""):
    ax.clear()
    ax.set_facecolor(WHITE)
    ax.set_aspect("equal")
    ax.axis("off")
    half = ref * 0.62
    ax.set_xlim(-half, half)
    ax.set_ylim(-half * 1.02, half * 1.16)

    # Reference square first: the footprint these molecules would occupy at the
    # measured density. Identical in every panel, so the boxes are comparable.
    ax.add_patch(plt.Rectangle((-ref / 2, -ref / 2), ref, ref, fill=False,
                               edgecolor=CRITICAL, lw=1.2, ls=(0, (4, 3)),
                               alpha=0.7, zorder=1))
    ax.add_patch(plt.Rectangle((-box / 2, -box / 2), box, box, fill=False,
                               edgecolor=INK_MUTED, lw=1.1, zorder=2))

    keep = slab_indices(pos, box, SLAB_NM)
    xy = (pos[keep][:, :2] % box) - box / 2.0
    radius = sigma / 2.0
    for x, y in xy:
        ax.add_patch(plt.Circle((x, y), radius, facecolor=colour, alpha=0.55,
                                edgecolor="white", lw=0.6, zorder=3))

    n_beads = pos.shape[0]
    phi = n_beads * (math.pi / 6.0) * sigma ** 3 / box ** 3
    rho = RHO_STAR * (ref / box) ** 3
    ax.set_title(label, fontsize=12.5, color=colour, loc="center", pad=4)
    if stamp:
        ax.text(0, half * 1.10, stamp, ha="center", va="top", fontsize=9.5,
                color=INK_MUTED)
    ax.text(0, -half * 0.86,
            f"box {box:.3f} nm    $\\rho$ {rho:.0f} kg m$^{{-3}}$ "
            f"({100 * (rho - RHO_STAR) / RHO_STAR:+.1f}%)\n"
            f"$\\sigma$ {sigma:.3f} nm    packing $\\phi$ {phi:.3f}    "
            f"{int(keep.sum())} beads in slab",
            ha="center", va="top", fontsize=10, color=INK_SECONDARY)


def main():
    if os.environ.get("MD_ENGINE") != "glycerol_real":
        raise SystemExit("run with MD_ENGINE=glycerol_real")
    if not SNAPSHOTS.exists():
        raise SystemExit(f"no snapshots at {SNAPSHOTS}; run make_real_box.py first")
    store = dict(np.load(SNAPSHOTS))
    ref = reference_box_nm()

    panels = [(n, c) for n, _d, c in ARM_SPECS if f"{n}|pos" in store]
    n_traj = min(store[f"{n}|pos"].shape[0] for n, _c in panels)
    picks = [0, n_traj // 2, n_traj - 1][:N_STATES]
    ps_per_frame = 60.0 / n_traj          # production is 60 ps
    labels = [f"{p * ps_per_frame:.0f} ps into production" for p in picks]
    labels[-1] = f"{picks[-1] * ps_per_frame:.0f} ps  ·  final state"

    fig, axes = plt.subplots(2, 2, figsize=(6.8, 7.2), facecolor=WHITE)
    header = fig.text(0.5, 0.985, "", ha="center", va="top", fontsize=10,
                      color=INK)

    def frame(k):
        idx = picks[k]
        for ax, (name, colour) in zip(axes.ravel(), panels):
            pos = store[f"{name}|pos"][idx]
            box = float(store[f"{name}|boxes"][idx])
            sigma = float(store[f"{name}|meta"][1])
            draw(ax, pos, box, sigma, colour, ref, name, stamp=labels[k])
        for ax in axes.ravel()[len(panels):]:
            ax.axis("off")
        header.set_text(
            f"a {SLAB_NM:.2f} nm slice through each fitted liquid   ·   seed {SEED}"
            f"   ·   beads at true $\\sigma/2$   ·   dashed square = experiment")
        return []

    frame(len(picks) - 1)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "glycerol-real-slab.png", dpi=200, facecolor=WHITE)
    print(f"wrote {OUT / 'glycerol-real-slab.png'}")

    anim = FuncAnimation(fig, frame, frames=len(picks), blit=False)
    _write_animation(anim, OUT / "glycerol-real-slab.webp", facecolor=WHITE,
                     frame_ms=FRAME_MS, hold_ms=HOLD_MS)
    for name, _c in panels:
        pos = store[f"{name}|pos"][-1]
        box = float(store[f"{name}|boxes"][-1])
        sigma = float(store[f"{name}|meta"][1])
        keep = slab_indices(pos, box, SLAB_NM)
        cover = keep.sum() * math.pi * (sigma / 2) ** 2 / box ** 2
        print(f"  {name:<24} {int(keep.sum()):>3} beads, coverage {cover:.2f}x")


if __name__ == "__main__":
    main()

"""A tilted 3D slice through each fitted liquid, at true bead size.

The whole-box view cannot show packing. Projecting 375 beads through a box that
is only ~7 beads deep gives an areal coverage of 6.1 to 6.7 times, so every
panel is six spheres deep and saturates: coverage grows as r^2/L^2 while packing
grows as r^3/L^3, compressing exactly the signal being looked for.

Thinning the slice fixes that, and the thickness is a real trade-off:

    0.35 nm   ~52 beads   17 molecules   coverage 0.9x
    0.50 nm   ~74 beads   25 molecules   coverage 1.25x     <- this figure
    1.00 nm  ~150 beads   50 molecules   coverage 2.5x
    full box  375 beads  125 molecules   coverage 6.4x

Coverage near 1.0 is where discs touch and gaps show. 0.50 nm keeps a single
bead of overlap, which under depth shading reads as depth rather than clutter,
so the slice looks three-dimensional without going back to a saturated blob.

The slice normal is world z and the camera is tilted 14 degrees off it, so the
slab is seen mostly face-on -- which is what keeps coverage near 1 -- while the
tilt makes its thickness visible. Beads are drawn at true sigma/2, sorted back
to front, faded and highlighted by depth.

Both outlines are slabs of the same thickness, so they compare like for like:
the solid one has the footprint of the box this arm reached, the dashed one the
footprint these 125 molecules would occupy at glycerol's measured density. The
gap between them is the density error, in the only units a picture has.

No simulation is re-run: this reads the trajectories make_real_box.py recorded.

    MD_ENGINE=glycerol_real python3 make_real_slab3d.py
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
    CRITICAL, INK, INK_MUTED, INK_SECONDARY, _write_animation,
)
from make_molecule_3d import AZIMUTH, _rx, _view  # noqa: E402
from make_real_box import (ARM_SPECS, RHO_STAR, SEED, SNAPSHOTS,  # noqa: E402
                          reference_box_nm)

OUT = Path("/Users/federico/Documents/personal/code/website/public/bayesopt-for-md-simulators")
WHITE = "#ffffff"
SLAB_NM = 0.50          # see the thickness table above
ELEVATION_DEG = -14.0   # matches the box figure, and tilts the slab off face-on
N_STATES = 3
FRAME_MS, HOLD_MS = 1100, 4000
BEADS_PER_MOLECULE = 3


def slab_cuboid_edges(side: float, thickness: float) -> list[np.ndarray]:
    """Edges of a box-footprint slab, centered on the origin."""
    hx, hz = side / 2.0, thickness / 2.0
    corners = np.array([[x, y, z] for x in (-hx, hx) for y in (-hx, hx)
                        for z in (-hz, hz)])
    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            if np.count_nonzero(np.abs(corners[i] - corners[j]) > 1e-9) == 1:
                edges.append(np.stack([corners[i], corners[j]]))
    return edges


def slab_floor_loop(side: float, thickness: float) -> np.ndarray:
    """The bottom face of a slab, closed, for drawing a footprint outline."""
    h, z = side / 2.0, -thickness / 2.0
    return np.array([[-h, -h, z], [h, -h, z], [h, h, z], [-h, h, z],
                     [-h, -h, z]])


def lighten(colour: str, amount: float = 0.55) -> tuple:
    r, g, b, _ = matplotlib.colors.to_rgba(colour)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def projected_bounds(view, sides, thickness, margin):
    """Shared limits: wide and short, so equal aspect leaves the axes wide."""
    xs, ys = [], []
    for side in sides:
        for e in slab_cuboid_edges(side, thickness):
            q = e @ view.T
            xs += list(q[:, 0]); ys += list(q[:, 1])
    return ((min(xs) - margin, max(xs) + margin),
            (min(ys) - margin, max(ys) + margin))


def draw(ax, pos, box, sigma, colour, ref, label, view, lims, stamp=""):
    ax.clear()
    ax.set_facecolor(WHITE)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(*lims[0])
    ax.set_ylim(*lims[1])

    # Same slab at the density experiment implies: identical in every panel, so
    # the arms stay comparable, and the same shape as the measured outline so
    # the two can be read against each other.
    # Only the floor face, and drawn over the beads: a full dashed wireframe
    # through the cloud is unreadable, and the footprint is what carries the
    # comparison anyway.
    q = slab_floor_loop(ref, SLAB_NM) @ view.T
    ax.plot(q[:, 0], q[:, 1], color=CRITICAL, lw=1.3, ls=(0, (3.5, 2.5)),
            alpha=0.85, zorder=6)
    # The slice actually drawn: its footprint is the box this arm reached.
    for e in slab_cuboid_edges(box, SLAB_NM):
        q = e @ view.T
        ax.plot(q[:, 0], q[:, 1], color=INK_MUTED, lw=1.2, alpha=0.9, zorder=2)

    centre = np.array([box] * 3) / 2.0
    wrapped = pos % box
    keep = np.abs(wrapped[:, 2] - box / 2.0) <= SLAB_NM / 2.0
    cam = (wrapped[keep] - centre) @ view.T
    cam = cam[np.argsort(cam[:, 2])]           # back to front

    radius = sigma / 2.0
    if len(cam):
        depth = cam[:, 2]
        span = max(depth.max() - depth.min(), 1e-9)
        shade = (depth - depth.min()) / span
    else:
        shade = np.empty(0)
    hi = lighten(colour)
    for (x, y, _z), f in zip(cam, shade):
        ax.add_patch(plt.Circle((x, y), radius, facecolor=colour,
                                alpha=0.42 + 0.40 * f, edgecolor="white",
                                lw=0.7, zorder=3))
        ax.add_patch(plt.Circle((x - 0.26 * radius, y + 0.26 * radius),
                                0.42 * radius, facecolor=hi,
                                alpha=0.30 + 0.35 * f, edgecolor="none",
                                zorder=4))

    n_slab = int(keep.sum())
    phi = pos.shape[0] * (math.pi / 6.0) * sigma ** 3 / box ** 3
    rho = RHO_STAR * (ref / box) ** 3
    ax.set_title(label, fontsize=12.5, color=colour, loc="center", pad=14)
    if stamp:
        ax.text(0.5, 1.015, stamp, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=9.5, color=INK_MUTED)
    ax.text(0.5, -0.03,
            f"box {box:.3f} nm    $\\rho$ {rho:.0f} "
            f"({100 * (rho - RHO_STAR) / RHO_STAR:+.1f}%)\n"
            f"$\\sigma$ {sigma:.3f} nm    $\\phi$ {phi:.3f}    "
            f"{n_slab // BEADS_PER_MOLECULE} molecules in slice",
            transform=ax.transAxes, ha="center", va="top", fontsize=9.5,
            color=INK_SECONDARY)


def main():
    if os.environ.get("MD_ENGINE") != "glycerol_real":
        raise SystemExit("run with MD_ENGINE=glycerol_real")
    if not SNAPSHOTS.exists():
        raise SystemExit(f"no snapshots at {SNAPSHOTS}; run make_real_box.py first")
    store = dict(np.load(SNAPSHOTS))
    ref = reference_box_nm()
    view = _view(AZIMUTH) @ _rx(math.radians(ELEVATION_DEG))

    panels = [(n, c) for n, _d, c in ARM_SPECS if f"{n}|pos" in store]
    n_traj = min(store[f"{n}|pos"].shape[0] for n, _c in panels)
    picks = [0, n_traj // 2, n_traj - 1][:N_STATES]
    ps_per_frame = 60.0 / n_traj          # production is 60 ps
    labels = [f"{p * ps_per_frame:.0f} ps into production" for p in picks]
    labels[-1] = f"{picks[-1] * ps_per_frame:.0f} ps  ·  final state"

    sides = [ref] + [float(store[f"{n}|boxes"].max()) for n, _c in panels]
    max_sigma = max(float(store[f"{n}|meta"][1]) for n, _c in panels)
    lims = projected_bounds(view, sides, SLAB_NM, max_sigma / 2.0 * 1.15)

    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.3), facecolor=WHITE)
    header = fig.text(0.5, 0.99, "", ha="center", va="top", fontsize=9.5,
                      color=INK, linespacing=1.5)

    def frame(k):
        idx = picks[k]
        for ax, (name, colour) in zip(axes.ravel(), panels):
            pos = store[f"{name}|pos"][idx]
            box = float(store[f"{name}|boxes"][idx])
            sigma = float(store[f"{name}|meta"][1])
            draw(ax, pos, box, sigma, colour, ref, name, view, lims,
                 stamp=labels[k])
        for ax in axes.ravel()[len(panels):]:
            ax.axis("off")
        header.set_text(
            f"a {SLAB_NM:.2f} nm slice through each fitted liquid   ·   seed {SEED}"
            f"   ·   beads at true $\\sigma/2$   ·   $\\rho$ in kg m$^{{-3}}$\n"
            f"solid outline = the box this arm reached   ·   "
            f"dashed floor = the footprint experiment implies")
        return []

    frame(len(picks) - 1)
    fig.tight_layout(rect=(0, 0.01, 1, 0.89), h_pad=1.6)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "glycerol-real-slab3d.png", dpi=200, facecolor=WHITE)
    print(f"wrote {OUT / 'glycerol-real-slab3d.png'}")

    anim = FuncAnimation(fig, frame, frames=len(picks), blit=False)
    _write_animation(anim, OUT / "glycerol-real-slab3d.webp", facecolor=WHITE,
                     frame_ms=FRAME_MS, hold_ms=HOLD_MS)
    for name, _c in panels:
        pos = store[f"{name}|pos"][-1]
        box = float(store[f"{name}|boxes"][-1])
        sigma = float(store[f"{name}|meta"][1])
        keep = np.abs((pos[:, 2] % box) - box / 2.0) <= SLAB_NM / 2.0
        cover = keep.sum() * math.pi * (sigma / 2) ** 2 / box ** 2
        print(f"  {name:<24} {int(keep.sum()):>3} beads "
              f"({int(keep.sum()) // BEADS_PER_MOLECULE} molecules), "
              f"coverage {cover:.2f}x")


if __name__ == "__main__":
    main()

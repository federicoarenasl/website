"""The molecule itself, redrawn at every parameter set the optimiser proposes.

The landscape animations show a point moving through an abstract slice of
parameter space. This one shows the thing that space describes. Four of the
nine parameters are directly geometric -- bond_length_nm and angle_deg are the
shape, sigma is the bead diameter -- and the two force constants are visible
through equipartition, which turns a stiffness into a thermal spread:

    bond:   sigma_r     = sqrt(kT / k_bond)
    angle:  sigma_theta = sqrt(kT / (k_angle * sin^2(theta0)))

The second follows from expanding V = 0.5 k (cos t - cos t0)^2 about t0, where
the curvature is k sin^2(t0) rather than k -- a cosine potential is softer at
its minimum than a harmonic one by exactly that factor, and stiff angles near
180 degrees are much floppier than their force constant suggests.

So each frame is an honest picture of the candidate molecule: its size, its
shape, and how much it rattles. The hidden reference is ghosted behind it.
Crashed proposals are drawn in red -- a shape that could not be simulated.

    python3 app/research/research/_bayesopt-md-scratch/make_molecule_animation.py
"""

import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation  # noqa: E402
from matplotlib.patches import Circle, Ellipse, Polygon  # noqa: E402

SCRATCH = Path(__file__).parent
sys.path.insert(0, str(SCRATCH))
from make_animation import (  # noqa: E402
    BASE, CRITICAL, INK, INK_MUTED, INK_SECONDARY, SURFACE, VIOLET,
    _write_animation, round_axes, round_corner_elbow, style_ax,
)

OUT = Path("/Users/federico/Documents/personal/code/website/public/bayesopt-for-md-simulators")
REPO = Path("/Users/federico/Documents/personal/code/agentic-optimiser")

K_B = 0.0083144621
TEMPERATURE_K = 298.15
KT = K_B * TEMPERATURE_K

TRUTH = {"epsilon": 2.7834, "sigma": 0.33470, "bond_length_nm": 0.31260,
         "bond_k": 2140.0, "angle_deg": 124.70, "angle_k": 38.60}
LOSS_FLOOR = 0.0351
THRESHOLD = 0.12
DPI = 130


def geometry(bond: float, angle_deg: float) -> np.ndarray:
    """Bead positions for a symmetric 3-bead molecule, vertex at the origin."""
    half = math.radians(angle_deg) / 2.0
    return np.array([[-bond * math.sin(half), bond * math.cos(half)],
                     [0.0, 0.0],
                     [bond * math.sin(half), bond * math.cos(half)]])


def thermal_spread(bond: float, bond_k: float, angle_deg: float, angle_k: float):
    """Radial and tangential standard deviations of the outer beads, in nm."""
    sigma_r = math.sqrt(KT / max(bond_k, 1e-9))
    sin2 = max(math.sin(math.radians(angle_deg)) ** 2, 1e-6)
    sigma_theta = math.sqrt(KT / (max(angle_k, 1e-9) * sin2))   # radians
    return sigma_r, bond * sigma_theta


def load(path: Path) -> list[dict]:
    rows = []
    for r in csv.DictReader(path.open()):
        rows.append({
            **{k: (float(r[k]) if r[k] else None) for k in
               ("epsilon", "sigma", "bond_length_nm", "bond_k", "angle_deg",
                "angle_k", "loss", "best_so_far")},
            "stable": r["stable"] == "True",
            "violations": r["violations"],
        })
    return rows


def build(csv_path: Path, out_name: str, label: str) -> None:
    rows = load(csv_path)
    n = len(rows)
    best = []
    cur = float("inf")
    for r in rows:
        if r["best_so_far"] is not None:
            cur = r["best_so_far"]
        best.append(cur)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.5), dpi=DPI,
                                  gridspec_kw={"width_ratios": [1.05, 1.0]})
    fig.patch.set_facecolor(BASE)
    for a in (ax, ax2):
        a.set_facecolor(SURFACE)

    # --- left: the molecule ------------------------------------------------
    lim = 0.42
    ax.set_xlim(-lim, lim)
    # Centred on the reference molecule's own extent rather than the origin,
    # which sits at the vertex bead and pushes the shape into the lower third.
    ax.set_ylim(-0.34, 0.50)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # Ghost of the hidden reference, drawn once and never updated.
    tpos = geometry(TRUTH["bond_length_nm"], TRUTH["angle_deg"])
    for i, j in ((0, 1), (1, 2)):
        ax.plot(*zip(tpos[i], tpos[j]), color=INK_MUTED, lw=1.2, ls=(0, (4, 3)),
                zorder=1, alpha=0.75)
    for p in tpos:
        ax.add_patch(Circle(p, TRUTH["sigma"] / 2, fill=False, lw=1.2,
                            ls=(0, (4, 3)), ec=INK_MUTED, alpha=0.75, zorder=1))

    bonds = [ax.plot([], [], lw=2.6, color=VIOLET, zorder=3, solid_capstyle="round")[0]
             for _ in range(2)]
    halos = [Ellipse((0, 0), 0, 0, facecolor=VIOLET, alpha=0.16, lw=0, zorder=2)
             for _ in range(2)]
    beads = [Circle((0, 0), 0.1, facecolor=VIOLET, ec=INK, lw=1.1, zorder=4)
             for _ in range(3)]
    for h in halos:
        ax.add_patch(h)
    for b in beads:
        ax.add_patch(b)

    caption = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", ha="left",
                      fontsize=8.5, color=INK_SECONDARY, family="monospace",
                      linespacing=1.5)

    # --- right: convergence ------------------------------------------------
    ax2.set_xlim(1, n)
    top = max(max(b for b in best if math.isfinite(b)), THRESHOLD * 1.4)
    ax2.set_ylim(0, top * 1.05)
    ax2.set_xlabel("Simulations run")
    ax2.set_ylabel("Best loss so far")
    style_ax(ax2)
    ax2.axhline(THRESHOLD, color=INK_MUTED, lw=1.0, ls="--", zorder=1)
    ax2.text(n * 0.97, THRESHOLD, "good enough", ha="right", va="bottom",
             fontsize=8, color=INK_MUTED)
    ax2.axhline(LOSS_FLOOR, color=INK_MUTED, lw=0.9, ls=":", zorder=1)
    ax2.text(n * 0.97, LOSS_FLOOR, "noise floor", ha="right", va="bottom",
             fontsize=8, color=INK_MUTED)
    curve, = ax2.plot([], [], color=VIOLET, lw=2.2, zorder=3)
    round_corner_elbow(ax2)

    def frame(k):
        r = rows[k]
        stable = r["stable"]
        colour = VIOLET if stable else CRITICAL
        pos = geometry(r["bond_length_nm"], r["angle_deg"])
        sr, st = thermal_spread(r["bond_length_nm"], r["bond_k"],
                                r["angle_deg"], r["angle_k"])

        for line, (i, j) in zip(bonds, ((0, 1), (1, 2))):
            line.set_data(*zip(pos[i], pos[j]))
            line.set_color(colour)
        for bead, p in zip(beads, pos):
            bead.center = p
            bead.set_radius(r["sigma"] / 2)
            bead.set_facecolor(colour)
            # Well depth as opacity: a deeper well is a stickier bead.
            bead.set_alpha(0.30 + 0.55 * min(r["epsilon"] / 8.0, 1.0))
        # Halos sit on the outer beads only; the vertex has no bond freedom.
        for halo, idx in zip(halos, (0, 2)):
            v = pos[idx] - pos[1]
            halo.set_center(pos[idx])
            # Clamped for display only. A near-linear angle has almost no
            # restoring curvature (it goes as sin^2), so the true tangential
            # spread diverges and would draw a spike off the panel. The
            # clamp caps how floppy the picture can look, not the physics.
            halo.width = 2 * min(st, 0.13)   # tangential, across the bond
            halo.height = 2 * sr          # radial, along the bond
            halo.angle = math.degrees(math.atan2(v[1], v[0])) - 90.0
            halo.set_facecolor(colour)

        status = "crashed: " + r["violations"] if not stable else f"loss {r['loss']:.3f}"
        caption.set_text(
            f"run {k + 1}/{n}   {status}\n"
            f"bond  {r['bond_length_nm']:.4f} nm   (truth {TRUTH['bond_length_nm']:.4f})\n"
            f"angle {r['angle_deg']:6.1f}°      (truth {TRUTH['angle_deg']:.1f})\n"
            f"sigma {r['sigma']:.4f} nm   (truth {TRUTH['sigma']:.4f})\n"
            f"eps   {r['epsilon']:.4f}      (truth {TRUTH['epsilon']:.4f})")

        xs = list(range(1, k + 2))
        ys = [b if math.isfinite(b) else None for b in best[:k + 1]]
        curve.set_data([x for x, y in zip(xs, ys) if y is not None],
                       [y for y in ys if y is not None])
        return [*bonds, *beads, *halos, curve, caption]

    fig.tight_layout()
    # The caption is deliberately not clipped: it sits near a rounded corner
    # and would lose its first characters to the curve.
    round_axes(ax, [*bonds, *beads, *halos])
    round_axes(ax2, [curve])
    anim = FuncAnimation(fig, frame, frames=n, blit=False)
    _write_animation(anim, OUT / out_name)
    plt.close(fig)


if __name__ == "__main__":
    src = sorted((REPO / "runs" / "glycerol" / "gp_untuned").glob("run_0/*/steps.csv"))
    if not src:
        raise SystemExit("no glycerol run found")
    build(src[0], "glycerol-molecule.gif", "GP (untuned)")

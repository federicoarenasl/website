"""The candidate molecule in 3D, rebuilt at every parameter set proposed.

The flat version encodes the same numbers, but a coarse-grained bead is a
sphere and reading two overlapping circles as two overlapping spheres asks the
viewer to do work the picture should do for them. Here each bead is shaded
from a light source, drawn back to front by depth, and the whole molecule spins
slowly so the geometry reads without a caption.

The hidden reference is drawn in the same scene as translucent grey spheres,
so convergence is visible directly: the coloured molecule grows, bends and
settles into the grey one, or fails to.

Spheres are composited as pre-shaded images rather than matplotlib 3D surfaces.
mplot3d resolves depth per-artist rather than per-pixel, so intersecting
surfaces are drawn in the wrong order; painting shaded discs back to front is
both correct for spheres and far faster.

    python3 app/projects/projects/_bayesopt-md-scratch/make_molecule_3d.py [arm]
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

SCRATCH = Path(__file__).parent
sys.path.insert(0, str(SCRATCH))
from make_animation import (  # noqa: E402
    BASE, CRITICAL, INK, INK_MUTED, INK_SECONDARY, SURFACE, VIOLET,
    _write_animation, round_axes, round_corner_elbow, style_ax,
)
from make_molecule_animation import (  # noqa: E402
    K_B, LOSS_FLOOR, TEMPERATURE_K, THRESHOLD, TRUTH, geometry, load,
    thermal_spread,
)

OUT = Path("/Users/federico/Documents/personal/code/website/public/bayesopt-for-md-simulators")
REPO = Path("/Users/federico/Documents/personal/code/agentic-optimiser")
DPI = 130
SPHERE_N = 96
# True isometric: the elevation at which the three world axes foreshorten
# equally is atan(1/sqrt(2)), and the azimuth is 45 degrees.
ELEVATION = math.radians(35.264)
AZIMUTH = math.radians(45.0)
# The molecule is planar, so it has no depth of its own -- what has depth is
# how its plane sits in space. Leaning it out of the view plane is what makes
# an isometric projection say anything at all here.
LEAN = math.radians(52.0)
# A small rock about the vertical: enough parallax to read as solid, not
# enough to lose the isometric character.
ROCK_AMPLITUDE = math.radians(11.0)
FLOOR_Y = -0.30
# Beads are drawn as an opaque core inside a translucent shell at the true
# Lennard-Jones radius. At this mapping sigma/2 is comparable to the bond
# length, so space-filling spheres merge into one blob -- the shell keeps the
# real size visible while the core shows the geometry.
CORE_FRACTION = 0.34


def _sphere_textures(n: int = SPHERE_N):
    """Shading and alpha for a unit sphere lit from the upper left.

    Returns the diffuse+ambient intensity and a coverage mask, both (n, n).
    The mask is feathered by one pixel at the rim: a hard edge on a small
    sphere aliases badly once it is scaled down into the frame.
    """
    u = np.linspace(-1.0, 1.0, n)
    x, y = np.meshgrid(u, -u)
    r2 = x * x + y * y
    inside = r2 <= 1.0
    z = np.sqrt(np.clip(1.0 - r2, 0.0, None))
    light = np.array([-0.5, 0.6, 0.62])
    light /= np.linalg.norm(light)
    lam = np.clip(x * light[0] + y * light[1] + z * light[2], 0.0, None)
    shade = 0.34 + 0.66 * lam ** 1.1          # ambient + diffuse
    shade += 0.30 * lam ** 14                 # narrow specular highlight
    edge = np.clip((1.0 - np.sqrt(r2)) * n / 2.0, 0.0, 1.0)
    return np.clip(shade, 0.0, 1.35), np.where(inside, edge, 0.0)


SHADE, MASK = _sphere_textures()


def _rgba(colour, shade, mask, alpha):
    """Tint the sphere shading with `colour` and return an RGBA image."""
    base = np.array(matplotlib.colors.to_rgb(colour))
    img = np.clip(base[None, None, :] * shade[:, :, None], 0.0, 1.0)
    out = np.zeros((*shade.shape, 4))
    out[..., :3] = img
    out[..., 3] = mask * alpha
    return out


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _view(azimuth: float) -> np.ndarray:
    """World -> camera. Screen is (x, y) of the result; z is depth to viewer.

    World y is up, so the azimuth turns about y and the elevation tilts the
    camera down onto the scene.
    """
    return _rx(-ELEVATION) @ _ry(azimuth)


def _draw_molecule(ax, centres, radius, colour, alpha, spheres):
    """Queue spheres for depth-sorted compositing; bonds are drawn as lines."""
    for c in centres:
        spheres.append((c[2], c, radius, colour, alpha))


def build(csv_path: Path, out_name: str) -> None:
    rows = load(csv_path)
    n = len(rows)
    best, cur = [], float("inf")
    for r in rows:
        if r["best_so_far"] is not None:
            cur = r["best_so_far"]
        best.append(cur)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.5), dpi=DPI,
                                  gridspec_kw={"width_ratios": [1.05, 1.0]})
    fig.patch.set_facecolor(BASE)
    for a in (ax, ax2):
        a.set_facecolor(SURFACE)

    lim = 0.62
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim * 0.80, lim * 0.80)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

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
    fig.tight_layout()
    round_axes(ax, [])
    round_axes(ax2, [curve])

    truth_xy = geometry(TRUTH["bond_length_nm"], TRUTH["angle_deg"])
    truth3 = np.column_stack([truth_xy[:, 0], truth_xy[:, 1], np.zeros(3)])
    truth3 -= truth3.mean(axis=0)

    def frame(k):
        for artist in list(ax.images) + list(ax.lines) + list(ax.texts) + list(ax.collections):
            artist.remove()

        r = rows[k]
        colour = VIOLET if r["stable"] else CRITICAL
        rock = ROCK_AMPLITUDE * math.sin(2 * math.pi * k / max(n - 1, 1))
        view = _view(AZIMUTH + rock)
        lean = _rx(LEAN)

        def to_screen(pts):
            cam = (view @ pts.T).T
            return cam[:, :2], cam[:, 2]

        # --- floor grid -------------------------------------------------
        # Depth in an isometric projection is ambiguous without a reference
        # surface: two beads at different depths project to the same place.
        # The grid and the drop lines below are what resolve it.
        g = 0.42
        floor_pts = []
        for t in np.linspace(-g, g, 7):
            floor_pts.append(np.array([[-g, FLOOR_Y, t], [g, FLOOR_Y, t]]))
            floor_pts.append(np.array([[t, FLOOR_Y, -g], [t, FLOOR_Y, g]]))
        for seg in floor_pts:
            xy, _ = to_screen(seg)
            ax.plot(xy[:, 0], xy[:, 1], color=INK_MUTED, lw=0.5, alpha=0.30, zorder=2)

        xy2 = geometry(r["bond_length_nm"], r["angle_deg"])
        cand = np.column_stack([xy2[:, 0], xy2[:, 1], np.zeros(3)])
        cand -= cand.mean(axis=0)
        cand = (lean @ cand.T).T

        truth_w = (lean @ truth3.T).T

        cand_s, cand_d = to_screen(cand)
        truth_s, truth_d = to_screen(truth_w)

        # --- drop lines and shadows -------------------------------------
        for i, c in enumerate(cand):
            foot = np.array([[c[0], FLOOR_Y, c[2]]])
            fs, _ = to_screen(foot)
            ax.plot([cand_s[i, 0], fs[0, 0]], [cand_s[i, 1], fs[0, 1]],
                    color=colour, lw=0.7, alpha=0.35, ls=(0, (2, 2)), zorder=3)
            ax.scatter(fs[0, 0], fs[0, 1], s=90, marker="o", color=INK_MUTED,
                       alpha=0.16, linewidths=0, zorder=3)

        queue = []
        for c, d in zip(truth_s, truth_d):
            queue.append((d, c, TRUTH["sigma"] / 2 * CORE_FRACTION, INK_MUTED, 0.55))
        for c, d in zip(cand_s, cand_d):
            alpha = 0.45 + 0.50 * min(r["epsilon"] / 8.0, 1.0)
            queue.append((d, c, r["sigma"] / 2 * CORE_FRACTION, colour, alpha))
            queue.append((d - 1e-6, c, r["sigma"] / 2, colour, 0.11))

        for i, j in ((0, 1), (1, 2)):
            ax.plot(*zip(truth_s[i], truth_s[j]), color=INK_MUTED, lw=1.3,
                    ls=(0, (4, 3)), alpha=0.5, solid_capstyle="round",
                    zorder=10 + (truth_d[i] + truth_d[j]) / 2 * 100)
            ax.plot(*zip(cand_s[i], cand_s[j]), color=colour, lw=3.0,
                    solid_capstyle="round",
                    zorder=10 + (cand_d[i] + cand_d[j]) / 2 * 100)

        for depth, c, rad, col, alpha in sorted(queue, key=lambda q: q[0]):
            ax.imshow(_rgba(col, SHADE, MASK, alpha),
                      extent=(c[0] - rad, c[0] + rad, c[1] - rad, c[1] + rad),
                      zorder=10 + depth * 100, interpolation="bilinear")

        status = ("crashed: " + r["violations"]) if not r["stable"] else f"loss {r['loss']:.3f}"
        ax.text(0.02, 0.97,
                f"run {k + 1}/{n}   {status}\n"
                f"bond  {r['bond_length_nm']:.4f} nm   (truth {TRUTH['bond_length_nm']:.4f})\n"
                f"angle {r['angle_deg']:6.1f}\u00b0      (truth {TRUTH['angle_deg']:.1f})\n"
                f"sigma {r['sigma']:.4f} nm   (truth {TRUTH['sigma']:.4f})\n"
                f"eps   {r['epsilon']:.4f}      (truth {TRUTH['epsilon']:.4f})",
                transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
                color=INK_SECONDARY, family="monospace", linespacing=1.5,
                zorder=200)

        xs = list(range(1, k + 2))
        pts = [(x, y) for x, y in zip(xs, best[:k + 1]) if math.isfinite(y)]
        curve.set_data([q[0] for q in pts], [q[1] for q in pts])
        return []

    anim = FuncAnimation(fig, frame, frames=n, blit=False)
    _write_animation(anim, OUT / out_name)
    plt.close(fig)


if __name__ == "__main__":
    arm = sys.argv[1] if len(sys.argv) > 1 else "gp_untuned"
    src = sorted((REPO / "runs" / "glycerol" / arm).glob("run_0/*/steps.csv"))
    if not src:
        raise SystemExit(f"no run found for arm {arm}")
    out_name = sys.argv[2] if len(sys.argv) > 2 else "glycerol-molecule-3d.gif"
    build(src[0], out_name)

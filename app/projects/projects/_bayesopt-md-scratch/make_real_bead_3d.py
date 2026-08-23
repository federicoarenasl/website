"""The bead in 3D, rebuilt at every proposal, scored against real glycerol.

Same renderer as the synthetic study's molecule figure -- pre-shaded spheres
composited back to front, isometric projection, a slow rock for parallax -- but
what it encodes is different, because what is being fitted is different.

In stage 1 the molecule's *shape* was the answer, so the figure showed a shape
converging on a grey ghost. Here the bonded geometry is frozen and the free
parameters are the bead's identity, so the shape never moves and three other
things do:

    sigma   -> the translucent shell, the bead's real Lennard-Jones size
    epsilon -> hue, from cool and weakly interacting to hot and sticky
    cutoff  -> the wire sphere, how far the attraction reaches before truncation

There is no ghost, because on an experimental target there is no true parameter
set to draw. The gauges take its place: measured density and enthalpy of
vaporization against the values real glycerol has.

    python3 make_real_bead_3d.py
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
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SCRATCH = Path(__file__).parent
sys.path.insert(0, str(SCRATCH))
WHITE = "#ffffff"   # matches the page, so the figure shows no panel edge

from make_animation import (  # noqa: E402
    AQUA, BASE, BLUE, CRITICAL, INK, INK_MUTED, INK_SECONDARY, ORANGE, SURFACE,
    VIOLET, _write_animation, style_ax,
)
from make_molecule_3d import (  # noqa: E402
    AZIMUTH, MASK, ROCK_AMPLITUDE, SHADE, _rgba, _rx, _view,
)

RUNS = Path("/Users/federico/Documents/personal/code/agentic-optimiser/runs")
OUT = Path("/Users/federico/Documents/personal/code/website/public/bayesopt-for-md-simulators")

BOND_NM, ANGLE_DEG = 0.3126, 124.7      # frozen: the shape never changes
LEAN = math.radians(52.0)
RHO_STAR, H_STAR = 1258.4, 91.0
# Hue spans the range the search actually occupies, not the declared bounds:
# over 1-8 every proposal worth looking at fell in the same brown, because they
# all live between 3.5 and 5.
EPS_RANGE = (3.2, 5.4)
# Smaller than the synthetic figure's 0.34: there the core carried the geometry,
# here the shell *is* sigma and should dominate.
CORE_FRACTION = 0.26
# Four arms on one seed. Seed 7 is the balanced pick: by combined log-deviation
# from each arm's own median it is where all four behave most typically at once.
SEED = 7
ARM_SPECS = [
    ("Bayesian Optimization (BO)", "glycerol_real", VIOLET),
    ("BO + feasibility model", "glycerol_real_gpfeas", AQUA),
    ("BO + LLM warm-start", "glycerol_real_llmwarm", BLUE),
    ("LLM", "glycerol_real_llm", ORANGE),
]

WELL_CMAP = LinearSegmentedColormap.from_list(
    "well", ["#b9c6d6", "#7fa4c6", "#e8a34a", "#e2622a", "#a02c18"])


def centres_3d():
    """The frozen three-bead geometry, leaned out of the view plane."""
    half = math.radians(ANGLE_DEG) / 2.0
    flat = np.array([[-BOND_NM * math.sin(half), BOND_NM * math.cos(half), 0.0],
                     [0.0, 0.0, 0.0],
                     [BOND_NM * math.sin(half), BOND_NM * math.cos(half), 0.0]])
    flat = flat - flat.mean(axis=0)
    return flat @ _rx(LEAN).T


def load_run(run_dir, seed):
    d = run_dir / f"run_{seed}"
    cands = sorted(d.glob("*/steps.csv"), key=lambda p: p.stat().st_size)
    if not cands:
        return []
    out = []
    for r in csv.DictReader(cands[-1].open()):
        loss = float(r["loss"]) if r["loss"] not in ("", None) else None
        dh = r.get("dhvap_kj_mol")
        out.append({
            "eps": float(r["epsilon"]), "sig": float(r["sigma"]),
            "cut": float(r["cutoff_nm"]), "loss": loss,
            "rho": float(r["density_kg_m3"]) if r.get("density_kg_m3") else None,
            "dhvap": float(dh) if dh not in (None, "") else None,
            "crashed": loss is None,
        })
    return out


def draw_scene(ax, step, k, n_frames):
    ax.clear()
    ax.set_facecolor(WHITE)
    ax.set_xlim(-0.62, 0.62)
    ax.set_ylim(-0.58, 0.46)
    ax.set_aspect("equal")
    ax.axis("off")
    if step is None:
        return

    azim = AZIMUTH + ROCK_AMPLITUDE * math.sin(2.0 * math.pi * k / max(n_frames, 1))
    view = _view(azim)
    cam = centres_3d() @ view.T
    frac = min(max((step["eps"] - EPS_RANGE[0]) / (EPS_RANGE[1] - EPS_RANGE[0]),
                   0.0), 1.0)
    hue = WELL_CMAP(frac)
    shell_r = step["sig"] / 2.0
    core_r = shell_r * CORE_FRACTION

    # Cutoff as a single equatorial ring about the molecule centre: the reach of
    # the attraction. Two rings at different tilts read as scribble rather than
    # as a sphere, so one is clearer than a wireframe.
    th = np.linspace(0, 2 * np.pi, 200)
    ring = np.stack([np.cos(th), np.sin(th), np.zeros_like(th)], axis=1)
    r3 = (ring * step["cut"] / 2.0) @ _rx(math.pi / 2).T @ view.T
    ax.plot(r3[:, 0], r3[:, 1], color=INK_MUTED, lw=0.9, alpha=0.55,
            ls=(0, (6, 4)), zorder=1)

    # Bonds behind the spheres; the shape is fixed so they never move.
    for a, b in ((0, 1), (1, 2)):
        ax.plot([cam[a, 0], cam[b, 0]], [cam[a, 1], cam[b, 1]], "-",
                color=INK_SECONDARY, lw=2.4, zorder=2, solid_capstyle="round")

    # Depth sort: translucent shells and opaque cores, painted back to front.
    queue = []
    for c in cam:
        queue.append((c[2] - 1e-3, c, shell_r, hue, 0.20 + 0.22 * frac))
        queue.append((c[2], c, core_r, hue, 1.0))
    for _z, c, r, colour, alpha in sorted(queue, key=lambda t: t[0]):
        ax.imshow(_rgba(colour, SHADE, MASK, alpha),
                  extent=(c[0] - r, c[0] + r, c[1] - r, c[1] + r),
                  zorder=3 + _z, interpolation="bilinear")

    label = (f"$\\epsilon$ {step['eps']:.2f}    $\\sigma$ {step['sig']:.3f}    "
             f"cutoff {step['cut']:.3f}")
    ax.text(0, -0.50, label, ha="center", va="center", fontsize=12,
            color=INK_SECONDARY)
    if step["crashed"]:
        ax.text(0, 0.40, "crashed", ha="center", va="center", fontsize=11,
                color=CRITICAL, style="italic")


def gauge(ax, step, key, target, lo, hi, label, unit, colour):
    ax.clear()
    ax.set_facecolor(WHITE)
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.tick_params(labelsize=9.5, colors=INK_MUTED)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(BASE)
    ax.axvline(target, color=CRITICAL, lw=1.6, zorder=3)
    ax.text(lo, 1.3, f"{label}", ha="left", va="bottom", fontsize=12,
            color=INK_SECONDARY)
    # Right-aligned rather than centered on the target line: centered, it sat on
    # top of the measured value's label whenever the fit was close -- which is
    # exactly when the reader most wants to compare the two numbers.
    ax.text(hi, 1.3, f"target {target:g} {unit}", ha="right", va="bottom",
            fontsize=10.5, color=CRITICAL)
    value = None if step is None else step.get(key)
    if value is None:
        # "crashed" and "never logged" are different failures and must not read
        # the same: one is the search's doing, the other is ours.
        msg = "crashed" if (step and step["crashed"]) else "not recorded"
        ax.text((lo + hi) / 2, 0.5, msg, ha="center", va="center",
                fontsize=10, color=INK_MUTED, style="italic")
        return
    ax.barh([0.5], [value - lo], left=lo, height=0.46, color=colour, alpha=0.5,
            zorder=2)
    ax.plot([value], [0.5], "o", ms=6, color=colour, markeredgecolor="white",
            markeredgewidth=0.8, zorder=4)
    txt = f"{value:.0f}" if abs(value) > 200 else f"{value:.1f}"
    ax.annotate(txt, (value, 0.78), xytext=(0, 2), textcoords="offset points",
                ha="center", va="bottom", fontsize=10, color=INK,
                fontweight="medium")



def gauge_shared(ax, entries, target, lo, hi, label, unit):
    """One property, every arm on the same scale.

    entries: list of (name, value_or_None, colour, crashed).
    """
    ax.clear()
    ax.set_facecolor(WHITE)
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.4, len(entries) - 0.4)
    ax.set_yticks([])
    ax.tick_params(labelsize=10, colors=INK_MUTED)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(BASE)
    ax.axvline(target, color=CRITICAL, lw=1.8, zorder=3)
    ax.text(lo, len(entries) - 0.25, label, ha="left", va="bottom",
            fontsize=11.5, color=INK_SECONDARY)
    ax.text(hi, len(entries) - 0.25, f"target {target:g} {unit}", ha="right",
            va="bottom", fontsize=10.5, color=CRITICAL)
    for i, (_name, value, colour, crashed) in enumerate(entries):
        y = len(entries) - 1 - i
        if value is None:
            ax.text((lo + hi) / 2, y, "crashed" if crashed else "not recorded",
                    ha="center", va="center", fontsize=9.5, color=INK_MUTED,
                    style="italic")
            continue
        ax.barh([y], [value - lo], left=lo, height=0.5, color=colour,
                alpha=0.5, zorder=2)
        ax.plot([value], [y], "o", ms=6, color=colour, markeredgecolor="white",
                markeredgewidth=0.8, zorder=4)
        txt = f"{value:.0f}" if abs(value) > 200 else f"{value:.1f}"
        ax.annotate(txt, (value, y), xytext=(6, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=10, color=INK)


def main():
    panels = [(name, load_run(RUNS / d, SEED), colour)
              for name, d, colour in ARM_SPECS]
    panels = [q for q in panels if q[1]]
    n_frames = max(len(t) for _n, t, _c in panels)

    ncol = 2
    nrow = (len(panels) + ncol - 1) // ncol
    fig = plt.figure(figsize=(6.6, 2.7 * nrow + 2.4), facecolor=WHITE)
    gs = fig.add_gridspec(nrow + 1, ncol, height_ratios=[3.0] * nrow + [2.4],
                          hspace=0.18, wspace=0.10)
    scene_axes = [fig.add_subplot(gs[i // ncol, i % ncol])
                  for i in range(len(panels))]
    gsub = gs[nrow, :].subgridspec(2, 1, hspace=0.75)
    g_rho = fig.add_subplot(gsub[0])
    g_h = fig.add_subplot(gsub[1])
    caption = fig.text(0.5, 0.99, "", fontsize=13, color=INK,
                       ha="center", va="top")

    def frame(k):
        entries_rho, entries_h = [], []
        for (name, traj, colour), sax in zip(panels, scene_axes):
            step = traj[k] if k < len(traj) else (traj[-1] if traj else None)
            draw_scene(sax, step, k, n_frames)
            sax.set_title(name, fontsize=12, color=colour, loc="center", pad=0)
            crashed = bool(step and step["crashed"])
            entries_rho.append((name, None if step is None else step["rho"],
                                colour, crashed))
            entries_h.append((name, None if step is None else step["dhvap"],
                              colour, crashed))
        gauge_shared(g_rho, entries_rho, RHO_STAR, 700, 1750, "density",
                     "kg m$^{-3}$")
        gauge_shared(g_h, entries_h, H_STAR, 20, 150, "$\\Delta H_{vap}$",
                     "kJ mol$^{-1}$")
        tag = "final" if k >= n_frames - 1 else f"simulation {k + 1}"
        caption.set_text(f"{tag}   ·   seed {SEED}")
        return []

    # Three states instead of forty: first proposal, halfway, final. The axis
    # here is the optimiser's step count, not simulation time, so these are
    # "where the search started, got to, and ended".
    picks = [0, n_frames // 2, n_frames - 1]

    def three(k):
        return frame(picks[k])

    three(len(picks) - 1)
    anim = FuncAnimation(fig, three, frames=len(picks), blit=False)
    OUT.mkdir(parents=True, exist_ok=True)
    _write_animation(anim, OUT / "glycerol-real-bead-3d.webp", facecolor=WHITE,
                     frame_ms=1100, hold_ms=4000)


if __name__ == "__main__":
    main()

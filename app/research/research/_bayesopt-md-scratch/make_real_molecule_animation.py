"""The bead, redrawn at every proposal.

Stage 1 could draw the candidate *molecule*, because four of the nine free
parameters were its geometry. Here geometry is frozen and what is being fitted
is the bead's identity, so that is what gets drawn:

    sigma   -> bead radius, to scale against the fixed 0.3126 nm bonds
    epsilon -> hue and halo, how hard this bead pulls on its neighbours
    cutoff  -> dashed ring, how far that pull reaches before truncation

There is no ghost of a true molecule to compare against, because on an
experimental target there is no true parameter set. What replaces it is the pair
of gauges: measured density and enthalpy of vaporisation against the values
real glycerol has. The shape stops being the answer and the properties become
the answer.

DHvap is greyed out for runs logged before the logger recorded it, rather than
reconstructed -- an inferred bar next to a measured one would read as the same
kind of evidence.

    python3 make_real_molecule_animation.py
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
from matplotlib.colors import LinearSegmentedColormap, to_rgba  # noqa: E402

SCRATCH = Path(__file__).parent
sys.path.insert(0, str(SCRATCH))
from make_animation import (  # noqa: E402
    BASE, CRITICAL, INK, INK_MUTED, INK_SECONDARY, ORANGE, SURFACE, VIOLET,
    _write_animation, style_ax,
)

RUNS = Path("/Users/federico/Documents/personal/code/agentic-optimiser/runs")
OUT = Path("/Users/federico/Documents/personal/code/website/public/bayesopt-for-md-simulators")

BOND_NM, ANGLE_DEG = 0.3126, 124.7      # frozen: the molecule does not change
RHO_STAR, H_STAR = 1258.4, 91.0
EPS_RANGE = (1.0, 8.0)
SEED = 0

# Cool-to-hot: a shallow well is a weakly interacting bead, a deep one is a
# sticky bead. Deliberately not the loss colormap -- this encodes a parameter,
# not a score, and reusing the loss ramp would imply hot means good.
WELL_CMAP = LinearSegmentedColormap.from_list(
    "well", ["#cfd8e3", "#8fb0d0", "#e8a34a", "#e2622a", "#a8301a"])


def bead_positions():
    """Two bonds meeting at the central bead, at the frozen equilibrium angle."""
    half = math.radians(ANGLE_DEG) / 2.0
    return np.array([[-BOND_NM * math.sin(half), BOND_NM * math.cos(half)],
                     [0.0, 0.0],
                     [BOND_NM * math.sin(half), BOND_NM * math.cos(half)]])


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
        })
    return out


def draw_bead_panel(ax, step, colour):
    """One frame of the bead drawing; returns nothing, redraws in place."""
    ax.clear()
    ax.set_facecolor(SURFACE)
    ax.set_xlim(-0.75, 0.75)
    ax.set_ylim(-0.62, 0.88)
    ax.set_aspect("equal")
    ax.axis("off")
    if step is None:
        return
    pos = bead_positions()
    centre = pos.mean(axis=0)
    frac = (step["eps"] - EPS_RANGE[0]) / (EPS_RANGE[1] - EPS_RANGE[0])
    hue = WELL_CMAP(min(max(frac, 0.0), 1.0))

    # Cutoff: how far this bead's attraction reaches, from the molecule centre.
    ax.add_patch(plt.Circle(centre, step["cut"] / 2.0, fill=False,
                            edgecolor=INK_MUTED, lw=0.9, ls=(0, (4, 4)),
                            alpha=0.75, zorder=1))
    for p in pos:
        # Halo: the well depth as reach and intensity, drawn as nested rings so
        # a deep well reads as a stronger pull rather than just a darker dot.
        for k, alpha in ((2.0, 0.10), (1.55, 0.16), (1.2, 0.22)):
            ax.add_patch(plt.Circle(p, step["sig"] / 2.0 * k, lw=0,
                                    facecolor=to_rgba(hue, alpha * (0.35 + frac)),
                                    zorder=2))
    for a, b in ((0, 1), (1, 2)):
        ax.plot([pos[a, 0], pos[b, 0]], [pos[a, 1], pos[b, 1]], "-",
                color=INK_SECONDARY, lw=2.0, zorder=3, solid_capstyle="round")
    for p in pos:
        ax.add_patch(plt.Circle(p, step["sig"] / 2.0, facecolor=hue,
                                edgecolor="white", lw=1.2, zorder=4))

    ax.text(0, -0.5, f"$\\epsilon$ {step['eps']:.2f}   "
                     f"$\\sigma$ {step['sig']:.3f}   "
                     f"cutoff {step['cut']:.3f}",
            ha="center", va="center", fontsize=8.5, color=INK_SECONDARY)


def gauge(ax, value, target, lo, hi, label, unit, colour):
    """A measured property against the value real glycerol has."""
    ax.clear()
    ax.set_facecolor(SURFACE)
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.tick_params(labelsize=7, colors=INK_MUTED)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(BASE)
    ax.axvline(target, color=CRITICAL, lw=1.6, zorder=3)
    ax.text(target, 1.12, f"{label}  {target:g} {unit}", ha="center",
            va="bottom", fontsize=7.5, color=CRITICAL)
    if value is None:
        ax.text((lo + hi) / 2, 0.5, "not recorded", ha="center", va="center",
                fontsize=7.5, color=INK_MUTED, style="italic")
        return
    ax.barh([0.5], [value - lo], left=lo, height=0.5, color=colour, alpha=0.55,
            zorder=2)
    ax.plot([value], [0.5], "o", ms=6, color=colour, markeredgecolor="white",
            markeredgewidth=0.8, zorder=4)
    ax.text(value, -0.42, f"{value:.0f}" if value > 200 else f"{value:.1f}",
            ha="center", va="top", fontsize=7.5, color=INK)


def main():
    candidates = [RUNS / "glycerol_real", RUNS / "glycerol_real_v1"]
    bo_dir = max(candidates, key=lambda d: len(load_run(d, SEED)))
    arms = [("Bayesian optimisation", load_run(bo_dir, SEED), VIOLET),
            ("LLM", load_run(RUNS / "glycerol_real_llm", SEED), ORANGE)]
    arms = [(n, t, c) for n, t, c in arms if t]
    n_frames = max(len(t) for _, t, _ in arms)

    fig = plt.figure(figsize=(10.0, 6.8), facecolor=SURFACE)
    gs = fig.add_gridspec(3, len(arms), height_ratios=[3.0, 1.5, 1.4],
                          hspace=0.95, wspace=0.22)
    bead_axes = [fig.add_subplot(gs[0, i]) for i in range(len(arms))]
    # Each arm gets two gauges stacked inside its cell. Built once: an earlier
    # version created a set of axes here and then rebuilt the list, leaving the
    # discarded axes on the figure to draw their default 0-1 frame underneath.
    gauge_axes = []
    for i in range(len(arms)):
        sub = gs[1, i].subgridspec(2, 1, hspace=2.6)
        gauge_axes.append((fig.add_subplot(sub[0]), fig.add_subplot(sub[1])))
    conv = fig.add_subplot(gs[2, :])

    conv.set_xlim(1, n_frames)
    conv.set_yscale("log")
    conv.set_xlabel("simulations spent", fontsize=8.5, color=INK_SECONDARY)
    conv.set_ylabel("best loss", fontsize=8.5, color=INK_SECONDARY)
    style_ax(conv, grid_axis="y")
    lows = [min([s["loss"] for s in t if s["loss"]] or [1]) for _, t, _ in arms]
    conv.set_ylim(min(lows) * 0.6, 1.3)
    lines = [conv.plot([], [], lw=2.0, color=c, label=n)[0] for n, _t, c in arms]
    conv.legend(frameon=False, fontsize=8, loc="upper right",
                labelcolor=INK_SECONDARY)
    caption = fig.text(0.012, 0.972, "", fontsize=9, color=INK)

    for ax, (name, _t, colour) in zip(bead_axes, arms):
        ax.set_title(name, fontsize=10, color=colour, loc="center", pad=4)

    def frame(k):
        for (name, traj, colour), bax, (g1, g2), line in zip(
                arms, bead_axes, gauge_axes, lines):
            step = traj[k] if k < len(traj) else (traj[-1] if traj else None)
            draw_bead_panel(bax, step, colour)
            bax.set_title(name, fontsize=10, color=colour, loc="center", pad=4)
            gauge(g1, step["rho"] if step else None, RHO_STAR, 700, 1750,
                  "density", "kg m$^{-3}$", colour)
            gauge(g2, step["dhvap"] if step else None, H_STAR, 20, 150,
                  "$\\Delta H_{vap}$", "kJ mol$^{-1}$", colour)
            best, xs, ys = math.inf, [], []
            for i, s in enumerate(traj[:k + 1], start=1):
                if s["loss"] is not None:
                    best = min(best, s["loss"])
                if math.isfinite(best):
                    xs.append(i)
                    ys.append(best)
            line.set_data(xs, ys)
        caption.set_text(f"simulation {min(k + 1, n_frames)}")
        return []

    anim = FuncAnimation(fig, frame, frames=n_frames, blit=False)
    OUT.mkdir(parents=True, exist_ok=True)
    _write_animation(anim, OUT / "glycerol-real-bead.gif")


if __name__ == "__main__":
    main()

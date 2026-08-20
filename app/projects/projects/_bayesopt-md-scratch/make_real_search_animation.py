"""How the search moves across the real-target landscape, step by step.

The static landscape shows where the good region is. What it cannot show is the
thing the comparison is actually about: how many simulations each method spends
before it gets there, and how much of the budget goes into the infeasible half
of the field.

So the field is the background and the proposals accumulate on it in the order
they were made, with a convergence trace underneath. BO and the LLM are drawn
side by side on the same field and the same clock.

Two honesties built into the drawing:

* The field is a slice at one cutoff. Every proposal has its own, so a marker's
  position on this map is exact in (epsilon, sigma) and approximate in the third
  dimension. Off-slice proposals are drawn faded rather than silently placed as
  though they belonged here.
* Crashes are drawn as crosses where they were proposed. They are part of what a
  budget buys, and leaving them out would make both searches look tidier than
  they were.

    python3 make_real_search_animation.py
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
WHITE = "#ffffff"   # matches the page, so the figure shows no panel edge

from make_animation import (  # noqa: E402
    BASE, CRITICAL, INK, INK_MUTED, INK_SECONDARY, LOSS_CMAP, ORANGE, SURFACE,
    VIOLET, _write_animation, style_ax,
)

LANDSCAPE = SCRATCH / "real_landscape.csv"
RUNS = Path("/Users/federico/Documents/personal/code/agentic-optimiser/runs")
OUT = Path("/Users/federico/Documents/personal/code/website/public/bayesopt-for-md-simulators")
CUTOFF = 1.1159
# The loss the best-known parameters actually achieve, measured on 16 seeds none
# of the optimisers saw. Not a theoretical bound: it is what the swept field
# minimum re-measures to once best-of-N optimism is removed, so anything below it
# is a luckier trajectory rather than a better force field.
NOISE_FLOOR = 0.0248
# One representative seed. With twelve LLM seeds against twelve BO seeds, this is
# the pair whose combined log-deviation from each arm's own median is smallest:
# LLM 0.0104 against its median 0.0093, BO 0.0451 against its 0.0732.
SEEDS = [4]


def load_field():
    rows = list(csv.DictReader(LANDSCAPE.open()))
    eps = sorted({float(r["epsilon"]) for r in rows})
    sig = sorted({float(r["sigma"]) for r in rows})
    grid = np.full((len(sig), len(eps)), np.nan)
    for r in rows:
        if r["loss"] != "":
            grid[sig.index(float(r["sigma"])), eps.index(float(r["epsilon"]))] = \
                float(r["loss"])
    return np.array(eps), np.array(sig), grid


def load_run(run_dir, seed):
    d = run_dir / f"run_{seed}"
    cands = sorted(d.glob("*/steps.csv"), key=lambda p: p.stat().st_size)
    if not cands:
        return []
    out = []
    for r in csv.DictReader(cands[-1].open()):
        loss = float(r["loss"]) if r["loss"] not in ("", None) else None
        out.append({"eps": float(r["epsilon"]), "sig": float(r["sigma"]),
                    "cut": float(r["cutoff_nm"]), "loss": loss})
    return out


def main():
    eps, sig, grid = load_field()
    # Prefer whichever BO directory has the longer trajectory: a re-run in
    # progress would otherwise be picked over a completed one and silently
    # truncate the comparison to however far it had got.
    candidates = [RUNS / "glycerol_real", RUNS / "glycerol_real_v1"]
    bo_dir = max(candidates, key=lambda d: len(load_run(d, SEEDS[0])))
    panels = []          # (row, column, label, trajectory, colour)
    for row, seed in enumerate(SEEDS):
        panels.append((row, 0, f"Bayesian optimisation · seed {seed}",
                       load_run(bo_dir, seed), VIOLET))
        panels.append((row, 1, f"LLM · seed {seed}",
                       load_run(RUNS / "glycerol_real_llm", seed), ORANGE))
    panels = [p for p in panels if p[3]]
    n_frames = max(len(p[3]) for p in panels)

    # Sized for a 672px reading column on a phone: a smaller canvas at the same
    # dpi makes every label larger relative to the figure, which is the only
    # thing that matters for legibility once it is scaled to fit.
    # Stacked, not side by side. Two columns on a phone give each panel about a
    # third of the screen, which no amount of type scaling rescues; one column
    # doubles the linear size of everything for the price of height, and height
    # is free when the reader scrolls.
    n_panels = len(panels)
    fig = plt.figure(figsize=(7.6, 6.2), facecolor=WHITE)
    gs = fig.add_gridspec(2, n_panels, height_ratios=[3.4, 1.9],
                          hspace=0.46, wspace=0.22)
    field_axes = [fig.add_subplot(gs[0, i]) for i in range(n_panels)]
    conv = fig.add_subplot(gs[1, :])
    conv.set_facecolor(WHITE)

    finite = grid[np.isfinite(grid)]
    levels = np.linspace(float(finite.min()), float(np.percentile(finite, 90)), 20)
    ei, si = np.meshgrid(eps, sig)
    crashed_cells = ~np.isfinite(grid)
    kmin = np.unravel_index(np.nanargmin(grid), grid.shape)

    for ax, (_r, _c, name, _t, colour) in zip(field_axes, panels):
        ax.set_facecolor("#f0efe9")
        ax.contourf(eps, sig, grid, levels=levels, cmap=LOSS_CMAP, extend="max")
        ax.plot(ei[crashed_cells], si[crashed_cells], "x", color=INK_MUTED,
                ms=3, alpha=0.45, mew=0.7)
        ax.plot([eps[kmin[1]]], [sig[kmin[0]]], marker="*", ms=14,
                color=CRITICAL, markeredgecolor="white", markeredgewidth=0.7,
                zorder=8)
        ax.set_title(name, fontsize=14, color=colour, loc="left", pad=8)
        ax.set_xlabel(r"$\epsilon$ (kJ mol$^{-1}$)", fontsize=12,
                      color=INK_SECONDARY)
        ax.tick_params(labelsize=11)
        style_ax(ax)
    field_axes[0].set_ylabel(r"$\sigma$ (nm)", fontsize=12, color=INK_SECONDARY)

    conv.set_xlim(1, n_frames)
    conv.set_yscale("log")
    conv.set_xlabel("simulations spent", fontsize=12, color=INK_SECONDARY)
    conv.set_ylabel("best loss", fontsize=12, color=INK_SECONDARY)
    conv.tick_params(labelsize=11)
    style_ax(conv, grid_axis="y")

    lows = [min([s["loss"] for s in t if s["loss"]] or [1]) for *_x, t, _c in panels]
    conv.set_ylim(min(min(lows), NOISE_FLOOR) * 0.55, 1.2)
    conv.axhline(NOISE_FLOOR, color=INK_MUTED, lw=1.1, ls=(0, (5, 3)), zorder=2)
    conv.text(n_frames * 0.985, NOISE_FLOOR * 1.10,
              f"achievable floor  {NOISE_FLOOR:.3f}", ha="right", va="bottom",
              fontsize=10, color=INK_MUTED)

    artists = []
    for ax, (_r, _c, _n, _t, colour) in zip(field_axes, panels):
        # Evaluated-but-not-improving proposals sit faint underneath: joining
        # every proposal draws the exploration, which is mostly noise. The path
        # connects only the moves that improved on the best so far, which is
        # the search's actual descent through the field.
        pts, = ax.plot([], [], "o", ms=4, color=colour, alpha=0.30, zorder=5)
        off, = ax.plot([], [], "o", ms=4, markerfacecolor="none",
                       markeredgecolor=colour, alpha=0.22, zorder=5)
        bad, = ax.plot([], [], "x", ms=6, color=INK, alpha=0.45, mew=1.1, zorder=5)
        path, = ax.plot([], [], "-o", ms=6, lw=1.8, color=colour, alpha=0.95,
                        markeredgecolor="white", markeredgewidth=0.7, zorder=7)
        cur, = ax.plot([], [], "o", ms=11, markerfacecolor="none",
                       markeredgecolor=INK, mew=1.4, zorder=9)
        artists.append((pts, off, bad, path, cur))
    # Dashed for the second seed, so four traces stay tellable apart by method
    # (colour) and by seed (dash) rather than needing four colours.
    lines = [conv.plot([], [], lw=2.0, color=c, label=n,
                       ls="-" if r == 0 else (0, (5, 2)))[0]
             for r, _c, n, _t, c in panels]
    conv.legend(frameon=False, fontsize=11, loc="upper right",
                labelcolor=INK_SECONDARY)
    caption = fig.text(0.5, 0.992, "", fontsize=13, color=INK,
                       ha="center", va="top")

    def frame(k):
        for (_r, _c, name, traj, colour), (pts, off, bad, path, cur), line in zip(
                panels, artists, lines):
            seen = traj[:k + 1]
            # Split by whether the proposal improved, and by whether its cutoff
            # puts it on this slice of the field.
            best_so_far, improving = math.inf, []
            for st in seen:
                if st["loss"] is not None and st["loss"] < best_so_far:
                    best_so_far = st["loss"]
                    improving.append((st["eps"], st["sig"]))
            improved = set(improving)
            good_on = [(st["eps"], st["sig"]) for st in seen
                       if st["loss"] is not None
                       and (st["eps"], st["sig"]) not in improved
                       and abs(st["cut"] - CUTOFF) <= 0.05]
            good_off = [(st["eps"], st["sig"]) for st in seen
                        if st["loss"] is not None
                        and (st["eps"], st["sig"]) not in improved
                        and abs(st["cut"] - CUTOFF) > 0.05]
            crashes = [(st["eps"], st["sig"]) for st in seen if st["loss"] is None]
            pts.set_data([q[0] for q in good_on], [q[1] for q in good_on])
            off.set_data([q[0] for q in good_off], [q[1] for q in good_off])
            bad.set_data([q[0] for q in crashes], [q[1] for q in crashes])
            path.set_data([q[0] for q in improving], [q[1] for q in improving])
            if seen:
                cur.set_data([seen[-1]["eps"]], [seen[-1]["sig"]])
            best, xs, ys = math.inf, [], []
            for i, s in enumerate(seen, start=1):
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
    _write_animation(anim, OUT / "glycerol-real-search.webp", facecolor=WHITE)


if __name__ == "__main__":
    main()

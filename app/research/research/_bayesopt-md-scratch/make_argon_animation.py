"""Animated GIFs of the argon parameter-recovery runs, on the real MD engine.

Same visual language as the surrogate animations, but the slice is now the two
constants being recovered — sigma and epsilon — and every point of the
background field is a real 11-second simulation rather than a formula.

    python3 app/research/research/_bayesopt-md-scratch/make_argon_animation.py

Writes into public/bayesopt-for-md-simulators/:
  argon-trajectory.gif            one tuned GP run
  argon-trajectory-warmstart.gif  random warm-up vs LLM warm-up
  argon-trajectory-three-arms.gif both, plus the LLM optimising alone
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.animation import FuncAnimation  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from pathlib import Path  # noqa: E402

from make_animation import (  # noqa: E402
    AQUA, BASE, BLUE, CRITICAL, INK, INK_MUTED, INK_SECONDARY, LOSS_CMAP,
    RUN_CMAP, SURFACE, VIOLET, YELLOW, _write_animation, round_axes,
    round_corner_elbow, style_ax,
)

SCRATCH = Path(__file__).parent
OUT = Path("/Users/federico/Documents/personal/code/website/public/bayesopt-for-md-simulators")

# Argon's accepted Lennard-Jones constants: the recovery target.
TRUE_SIGMA, TRUE_EPSILON = 0.3405, 0.9961
# Measured by repeating the true parameters across six seeds.
LOSS_FLOOR = 0.0365
THRESHOLD = 0.10
N_WARMUP = 10


def _load(name):
    df = pd.read_csv(SCRATCH / name)
    df["loss"] = pd.to_numeric(df["loss"], errors="coerce")
    df["best_so_far"] = pd.to_numeric(df["best_so_far"], errors="coerce")
    return df


def _draw_landscape(ax):
    # Seed-averaged on the GPU: each cell is the mean of five independent runs,
    # not one. A single run's loss carries a ~0.037 noise floor, which pocked the
    # earlier field with islands that read as secondary minima and put the
    # apparent optimum 10% off argon's true epsilon. Averaging fixes both.
    # The field is float32 (Metal has no float64) while the trajectories drawn
    # over it are float64: a 400-point paired comparison put the two at Pearson
    # r = 1.0000 with zero mean offset, so the mean of GPU runs is an unbiased
    # estimate of the same field the CPU trajectories were scored against.
    land = pd.read_csv(SCRATCH / "argon_landscape_gpu.csv")
    land["loss"] = pd.to_numeric(land["loss"], errors="coerce")
    eps = np.sort(land["epsilon"].unique())
    sig = np.sort(land["sigma"].unique())
    grid = land.pivot(index="epsilon", columns="sigma", values="loss").values
    # A cell counts as crashed when most of its seeds blew up, which puts the
    # boundary where the majority of runs fail rather than where one did.
    frac = land.pivot(index="epsilon", columns="sigma", values="crash_frac").values
    crashed = np.isnan(grid) | (frac > 0.5)

    ax.pcolormesh(sig, eps, np.where(crashed, 1.0, np.nan),
                  cmap=LinearSegmentedColormap.from_list("x", [BASE, BASE]),
                  shading="auto", zorder=1)
    hatch = ax.contourf(sig, eps, crashed.astype(float), levels=[0.5, 1.5],
                        colors="none", hatches=["////"], zorder=2)
    levels = np.arange(0.05, 0.7501, 0.0175)
    filled = ax.contourf(sig, eps, np.clip(grid, None, levels[-1]), levels=levels,
                         cmap=LOSS_CMAP, zorder=3)
    for width, colour in ((3, SURFACE), (1.5, INK)):
        ax.plot([TRUE_SIGMA], [TRUE_EPSILON], marker="o", markersize=9,
                markerfacecolor="none", markeredgecolor=colour, markeredgewidth=width,
                zorder=6)
    ax.set_xlim(sig.min(), sig.max())
    ax.set_ylim(eps.min(), eps.max())
    ax.set_xlabel("Lennard-Jones diameter σ (nm)")
    ax.set_ylabel("Lennard-Jones well depth ε (kJ/mol)")
    style_ax(ax)
    return filled, levels, [hatch, filled]


def build(arms, out_name):
    runs = {label: _load(f) for label, f, _ in arms}
    n_runs = int(max(df["step"].max() for df in runs.values()))
    single = len(arms) == 1

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.5),
                                  gridspec_kw={"width_ratios": [1.4, 1]})
    filled, levels, field = _draw_landscape(ax)

    cbar = fig.colorbar(filled, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("Loss vs measured properties", color=INK_SECONDARY, fontsize=9)
    cbar.outline.set_visible(False)
    cbar.set_ticks(levels[::10])
    cbar.ax.set_yticklabels([f"{v:.2f}" for v in levels[::10]])
    cbar.ax.tick_params(labelsize=8, color=INK_MUTED, labelcolor=INK_MUTED)

    if not single:
        ax2.axvspan(1, N_WARMUP, color=BASE, alpha=0.18, linewidth=0, zorder=1)
        ax2.annotate("warm-up", xy=(N_WARMUP / 2, 0.94), xycoords=("data", "axes fraction"),
                     fontsize=8, color=INK_SECONDARY, ha="center", zorder=2)
    ax2.axhline(THRESHOLD, color=INK_MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
    ax2.axhline(LOSS_FLOOR, color=BASE, linewidth=1, linestyle=(0, (2, 2)), zorder=2)
    ax2.annotate("good enough (0.10)", xy=(1, THRESHOLD), xytext=(0, 4),
                 textcoords="offset points", fontsize=8, color=INK_MUTED)
    ax2.annotate("noise floor (0.037)", xy=(1, LOSS_FLOOR), xytext=(0, -11),
                 textcoords="offset points", fontsize=8, color=INK_MUTED)
    ax2.set_xlim(1, n_runs)
    ax2.set_ylim(0.0, max(df["best_so_far"].max() for df in runs.values()) * 1.05)
    ax2.set_xlabel("Simulations run")
    ax2.set_ylabel("Best loss so far")
    style_ax(ax2, grid_axis="y")

    empty = np.empty((0, 2))
    artists = {}
    for i, (label, _, colour) in enumerate(arms):
        artists[label] = {
            "past": ax.scatter([], [], s=26, facecolor=INK_MUTED if single else colour,
                               edgecolor="none", alpha=0.5 if single else 0.35, zorder=8),
            "crashes": ax.scatter([], [], s=80, marker="x", color=CRITICAL,
                                  linewidth=1.7, zorder=11),
            "chain": ax.plot([], [], color=colour, linewidth=1.6, alpha=0.8, zorder=9)[0],
            "chain_pts": (ax.scatter([], [], s=70, c=[], cmap=RUN_CMAP, vmin=1, vmax=n_runs,
                                     edgecolor=SURFACE, linewidth=1.0, zorder=10)
                          if single else
                          ax.scatter([], [], s=64, facecolor=colour, edgecolor=SURFACE,
                                     linewidth=1.0, zorder=10)),
            "latest": ax.scatter([], [], s=170, facecolor="none", edgecolor=colour,
                                 linewidth=2.0, zorder=13),
            "star": ax.scatter([], [], s=250, marker="*",
                               facecolor=YELLOW if single else colour,
                               edgecolor=INK if single else SURFACE,
                               linewidth=0.9 if single else 1.4, zorder=12),
            "curve": ax2.plot([], [], color=colour, linewidth=2.0, zorder=5)[0],
            "head": ax2.scatter([], [], s=55, facecolor=colour, edgecolor=SURFACE,
                                linewidth=1.2, zorder=6),
        }
        if not single:
            ax2.annotate(label, xy=(0.97, 0.95 - 0.085 * i), xycoords="axes fraction",
                         fontsize=9, color=colour, fontweight="bold", ha="right")

    readout = ax.text(0.025, 0.965, "", transform=ax.transAxes, fontsize=9.5, color=INK,
                      va="top", ha="left", fontweight="bold", zorder=14, linespacing=1.4,
                      bbox=dict(boxstyle="round,pad=0.32", facecolor=SURFACE,
                                edgecolor="none", alpha=0.9))

    def update(frame):
        i = min(frame, n_runs)
        lines = [f"run {i}/{n_runs}"]
        for label, _, _c in arms:
            df, a = runs[label], artists[label]
            seen = df[df["step"] <= i]
            stable = seen[seen["loss"].notna()]
            dead = seen[seen["loss"].isna()]
            a["past"].set_offsets(seen[["sigma", "epsilon"]].to_numpy())
            a["crashes"].set_offsets(dead[["sigma", "epsilon"]].to_numpy() if len(dead) else empty)
            improving = stable[stable["loss"].cummin() == stable["loss"]]
            if len(improving):
                a["chain"].set_data(improving["sigma"], improving["epsilon"])
                a["chain_pts"].set_offsets(improving[["sigma", "epsilon"]].to_numpy())
                if single:
                    a["chain_pts"].set_array(improving["step"].to_numpy())
                best = improving.iloc[-1]
                a["star"].set_offsets([[best["sigma"], best["epsilon"]]])
                lines.append(f"best loss {best['loss']:.3f}" if single
                             else f"{label}: {best['loss']:.3f}")
            else:
                a["chain"].set_data([], [])
                a["chain_pts"].set_offsets(empty)
                a["star"].set_offsets(empty)
                lines.append("no stable run yet" if single else f"{label}: --")
            cur = seen.iloc[-1]
            a["latest"].set_offsets([[cur["sigma"], cur["epsilon"]]])
            a["curve"].set_data(seen["step"], seen["best_so_far"])
            tail = seen["best_so_far"].to_numpy()
            a["head"].set_offsets([[i, tail[-1]]] if not np.isnan(tail[-1]) else empty)
        readout.set_text("   ".join(lines) if single else "\n".join(lines))
        return []

    fig.tight_layout()
    on_landscape = ("past", "crashes", "chain", "chain_pts", "latest", "star")
    round_axes(ax, field + [artists[l][k] for l, _, _ in arms for k in on_landscape])
    round_axes(cbar.ax, [cbar.solids], radius_in=0.045)
    round_corner_elbow(ax2)

    anim = FuncAnimation(fig, update, frames=list(range(1, n_runs + 1)), blit=False)
    _write_animation(anim, OUT / out_name)


def main():
    build([("GP", "argon_random_warm.csv", BLUE)], "argon-trajectory.gif")
    build([("Random warm-up", "argon_random_warm.csv", BLUE),
           ("LLM warm-up", "argon_llm_warm.csv", AQUA)],
          "argon-trajectory-warmstart.gif")
    build([("Random warm-up", "argon_random_warm.csv", BLUE),
           ("LLM warm-up", "argon_llm_warm.csv", AQUA),
           ("LLM only", "argon_llm_only.csv", VIOLET)],
          "argon-trajectory-three-arms.gif")


if __name__ == "__main__":
    main()

"""Animated GIFs of Bayesian-optimization runs walking the loss landscape.

Run with a python that has matplotlib + pandas + Pillow:

    python3 app/research/research/_bayesopt-md-scratch/make_animation.py

Writes into public/bayesopt-for-md-simulators/:
  trajectory.gif            one tuned GP run (the shared seed)
  trajectory-warmstart.gif  random warm-up vs LLM warm-up, same seed, overlaid

The background uses banded contours rather than a continuous mesh: a GIF has a
256-colour palette, and a smooth gradient bands badly under quantisation while
discrete levels quantise cleanly and stay small.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from PIL import Image, ImageSequence  # noqa: E402
from pathlib import Path  # noqa: E402

SCRATCH = Path(__file__).parent
OUT = Path("/Users/federico/Documents/personal/code/website/public/bayesopt-for-md-simulators")

SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK_SECONDARY = "#52514e"; INK_MUTED = "#898781"
GRID = "#e1e0d9"; BASE = "#c3c2b7"
BLUE = "#2a78d6"; ORANGE = "#eb6834"; AQUA = "#1baf7a"; YELLOW = "#eda100"; CRITICAL = "#d03b3b"
# Third arm: validated against BLUE and AQUA for CVD separation, and chosen
# over the house orange, which would camouflage against the loss field.
VIOLET = "#7c3aed"

LOSS_CMAP = LinearSegmentedColormap.from_list(
    "loss", ["#fdf1e7", "#f9d3b4", "#f2ab77", "#eb6834", "#b8451c", "#7d2c10", "#4a1a09"]
)
RUN_CMAP = LinearSegmentedColormap.from_list(
    "run", ["#d3e3f7", "#a8c8ee", "#6ea3e0", "#2a78d6", "#1c548f", "#0f3355"]
)

LOSS_FLOOR = 0.1465
THRESHOLD = 0.18
OPTIMUM_DT, OPTIMUM_EPS = 19.16, 3.1006
N_WARMUP = 10

# 130 dpi renders 1430x585, about 2.1x the 672 px the figures display at.
# Counter-intuitively this is no larger than 92 dpi as a GIF: the loss field is
# flat colour bands, and at low dpi antialiasing blends them into many
# near-duplicate colours that dither badly under a 256-colour palette.
DPI = 130
FRAME_MS = 330        # per-run frame duration
FINAL_HOLD_MS = 2200  # linger on the final state before the loop restarts

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": BASE, "axes.labelcolor": INK_SECONDARY,
    "text.color": INK, "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
    "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 10,
})


CORNER_IN = 0.085  # corner radius, inches


def _axes_radii(ax, radius_in=CORNER_IN):
    """Corner radius in axes fractions, separately per axis.

    Axes-fraction units are not isotropic, so a single value would draw oval
    corners on a non-square panel. Deriving both from the axes' size in inches
    keeps the corner circular, and is independent of the save dpi.
    """
    pos = ax.get_position()
    fig_w, fig_h = ax.figure.get_size_inches()
    rx = radius_in / (pos.width * fig_w)
    ry = radius_in / (pos.height * fig_h)
    # A radius past half the span would fold the corners into each other; the
    # colourbar is narrow enough for this to bite.
    return min(rx, 0.5), min(ry, 0.5)


def _rounded_rect(rx, ry):
    """Closed rounded-rectangle path spanning the unit square."""
    P = MplPath
    verts = [(rx, 0), (1 - rx, 0), (1, 0), (1, ry), (1, 1 - ry), (1, 1), (1 - rx, 1),
             (rx, 1), (0, 1), (0, 1 - ry), (0, ry), (0, 0), (rx, 0), (rx, 0)]
    codes = [P.MOVETO, P.LINETO, P.CURVE3, P.CURVE3, P.LINETO, P.CURVE3, P.CURVE3,
             P.LINETO, P.CURVE3, P.CURVE3, P.LINETO, P.CURVE3, P.CURVE3, P.CLOSEPOLY]
    return P(verts, codes)


def round_axes(ax, artists, edge=BASE, radius_in=CORNER_IN):
    """Clip `artists` to a rounded rectangle and draw it as the panel border."""
    rx, ry = _axes_radii(ax, radius_in)
    patch = PathPatch(_rounded_rect(rx, ry), transform=ax.transAxes,
                      facecolor="none", edgecolor=edge, linewidth=1.0, zorder=20)
    ax.add_patch(patch)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for artist in artists:
        if artist is not None:
            artist.set_clip_path(patch)
    return patch


def round_corner_elbow(ax, edge=BASE):
    """Replace the left/bottom spine join with a single curved elbow."""
    rx, ry = _axes_radii(ax)
    P = MplPath
    path = P([(0, 1), (0, ry), (0, 0), (rx, 0), (1, 0)],
             [P.MOVETO, P.LINETO, P.CURVE3, P.CURVE3, P.LINETO])
    ax.add_patch(PathPatch(path, transform=ax.transAxes, facecolor="none",
                           edgecolor=edge, linewidth=1.0, clip_on=False, zorder=20))
    for spine in ax.spines.values():
        spine.set_visible(False)


def style_ax(ax, grid_axis=None):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)
        ax.spines[s].set_linewidth(1)
    ax.grid(False)
    if grid_axis:
        ax.grid(axis=grid_axis)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)


def _load(filename):
    df = pd.read_csv(SCRATCH / filename)
    df["loss"] = pd.to_numeric(df["loss"], errors="coerce")
    df["best_so_far"] = pd.to_numeric(df["best_so_far"], errors="coerce")
    return df


def _draw_landscape(ax):
    """Static background: the loss field, the crash region, the cliff, the optimum."""
    land = pd.read_csv(SCRATCH / "landscape.csv")
    boundary = pd.read_csv(SCRATCH / "stability_boundary.csv")
    eps_vals = np.sort(land["epsilon"].unique())
    dt_vals = np.sort(land["timestep_fs"].unique())
    grid = land.pivot(index="epsilon", columns="timestep_fs", values="loss").values
    crashed = np.isnan(grid)

    crash_mesh = ax.pcolormesh(dt_vals, eps_vals, np.where(crashed, 1.0, np.nan),
                               cmap=LinearSegmentedColormap.from_list("x", [BASE, BASE]),
                               shading="auto", zorder=1)
    hatch = ax.contourf(dt_vals, eps_vals, crashed.astype(float), levels=[0.5, 1.5],
                        colors="none", hatches=["////"], zorder=2)
    # 0.01-wide bands read as a smooth ramp while staying flat-shaded, which a
    # 256-colour GIF palette handles far better than a continuous gradient.
    # The field is clipped to the top level instead of extending it, so the
    # colourbar needs no arrow: losses above 0.55 saturate at the darkest step.
    levels = np.arange(0.15, 0.5501, 0.01)
    filled = ax.contourf(dt_vals, eps_vals, np.clip(grid, None, levels[-1]),
                         levels=levels, cmap=LOSS_CMAP, zorder=3)
    cliff, = ax.plot(boundary["max_stable_timestep_fs"], boundary["epsilon"],
                     color=CRITICAL, linewidth=1.6, zorder=5)
    for width, colour in ((3, SURFACE), (1.5, INK)):
        ax.plot([OPTIMUM_DT], [OPTIMUM_EPS], marker="o", markersize=9,
                markerfacecolor="none", markeredgecolor=colour, markeredgewidth=width,
                zorder=6)
    ax.set_xlim(dt_vals.min(), dt_vals.max())
    ax.set_ylim(eps_vals.min(), eps_vals.max())
    ax.set_xlabel("Timestep (fs)")
    ax.set_ylabel("Lennard-Jones well depth ε (kJ/mol)")
    style_ax(ax)
    return filled, levels, [crash_mesh, hatch, filled, cliff]


def _write_animation(anim, out, webp_quality=80, transparent=False,
                     facecolor=SURFACE, frame_ms=None, hold_ms=None):
    """Render frames, then re-save with per-frame durations.

    Pillow infers the container from the extension, so `.webp` yields an
    animated WebP: full colour and roughly a third smaller than the same frames
    as GIF, which is capped at a 256-colour palette. The re-save is needed
    either way because Pillow drops repeated identical frames, so the end pause
    has to be a per-frame duration rather than duplicated frames.
    """
    # transparent=True keeps the page background showing through. Only worth it
    # for .webp: GIF carries a single fully-transparent palette index, so every
    # antialiased edge -- all the text, every sphere rim -- has to snap to either
    # opaque or invisible, and fringes badly.
    # facecolor="none" rather than transparent=True: the latter also clears every
    # *axes* patch, which erases deliberate backgrounds -- the pale ground marking
    # the infeasible region, for one -- and makes "crashed" look identical to
    # "off the edge of the data".
    save_kwargs = ({"facecolor": "none"} if transparent
                   else {"facecolor": facecolor})
    anim.save(out, writer=PillowWriter(fps=1000 / FRAME_MS), dpi=DPI,
              savefig_kwargs=save_kwargs)
    plt.close("all")

    src = Image.open(out)
    seq = [f.copy() for f in ImageSequence.Iterator(src)]
    if transparent:
        seq = [f.convert("RGBA") for f in seq]
    # Overridable because a three-state figure wants long, readable holds where
    # a forty-step run wants a flick-book.
    per_frame = FRAME_MS if frame_ms is None else frame_ms
    durations = [per_frame] * len(seq)
    durations[-1] = FINAL_HOLD_MS if hold_ms is None else hold_ms
    common = dict(save_all=True, append_images=seq[1:], duration=durations, loop=0)
    if out.suffix == ".webp":
        # method=6 is the slowest, smallest setting; quality is the lossy knob.
        seq[0].save(out, quality=webp_quality, method=6, **common)
    else:
        # disposal=1 keeps each frame on the canvas so only changed pixels are
        # stored; every frame here adds to the picture, so nothing needs clearing.
        seq[0].save(out, disposal=1, optimize=True, **common)
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB, {len(seq)} frames, "
          f"{seq[0].size[0]}x{seq[0].size[1]})")


def build(arms: list[tuple[str, str, str]], out_name: str) -> None:
    """Animate one or more runs over the same landscape.

    Parameters
    ----------
    arms : list of (label, trajectory csv, colour)
        One entry animates a single run with its points shaded by run order.
        Two or more animate them together, each in its own colour.
    """
    runs = {label: _load(filename) for label, filename, _ in arms}
    n_runs = int(max(df["step"].max() for df in runs.values()))
    single = len(arms) == 1

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(11.0, 4.5), gridspec_kw={"width_ratios": [1.4, 1]}
    )
    filled, levels, field = _draw_landscape(ax)

    cbar = fig.colorbar(filled, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("Loss vs experiment", color=INK_SECONDARY, fontsize=9)
    cbar.outline.set_visible(False)
    cbar.set_ticks(levels[::10])
    cbar.ax.set_yticklabels([f"{v:.2f}" for v in levels[::10]])
    cbar.ax.tick_params(labelsize=8, color=INK_MUTED, labelcolor=INK_MUTED)

    if not single:
        ax2.axvspan(1, N_WARMUP, color=BASE, alpha=0.18, linewidth=0, zorder=1)
        # Axes fraction, not data coords: the y limit depends on the run.
        ax2.annotate("warm-up", xy=(N_WARMUP / 2, 0.94), xycoords=("data", "axes fraction"),
                     fontsize=8, color=INK_SECONDARY, ha="center", zorder=2)
    ax2.axhline(THRESHOLD, color=INK_MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
    ax2.axhline(LOSS_FLOOR, color=BASE, linewidth=1, linestyle=(0, (2, 2)), zorder=2)
    ax2.annotate("good enough (0.18)", xy=(1, THRESHOLD), xytext=(0, 4),
                 textcoords="offset points", fontsize=8, color=INK_MUTED)
    ax2.set_xlim(1, n_runs)
    ax2.set_ylim(0.13, max(df["best_so_far"].max() for df in runs.values()) * 1.03)
    ax2.set_xlabel("Simulations run")
    ax2.set_ylabel("Best loss so far")
    style_ax(ax2, grid_axis="y")

    empty = np.empty((0, 2))
    artists = {}
    for label, _, colour in arms:
        artists[label] = {
            "past": ax.scatter([], [], s=26,
                               facecolor=INK_MUTED if single else colour,
                               edgecolor="none", alpha=0.5 if single else 0.35, zorder=8),
            "crashes": ax.scatter([], [], s=80, marker="x", color=CRITICAL,
                                  linewidth=1.7, zorder=11),
            "chain": ax.plot([], [], color=colour, linewidth=1.6, alpha=0.8, zorder=9)[0],
            "chain_pts": (
                ax.scatter([], [], s=70, c=[], cmap=RUN_CMAP, vmin=1, vmax=n_runs,
                           edgecolor=SURFACE, linewidth=1.0, zorder=10)
                if single else
                ax.scatter([], [], s=64, facecolor=colour, edgecolor=SURFACE,
                           linewidth=1.0, zorder=10)
            ),
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
            ax2.annotate(label, xy=(0.97, 0.95 - 0.085 * list(runs).index(label)),
                         xycoords="axes fraction", fontsize=9, color=colour,
                         fontweight="bold", ha="right")

    readout = ax.text(0.025, 0.965, "", transform=ax.transAxes, fontsize=9.5, color=INK,
                      va="top", ha="left", fontweight="bold", zorder=14, linespacing=1.4,
                      bbox=dict(boxstyle="round,pad=0.32", facecolor=SURFACE,
                                edgecolor="none", alpha=0.9))

    def update(frame):
        i = min(frame, n_runs)
        lines = [f"run {i}/{n_runs}"]
        for label, _, _colour in arms:
            df = runs[label]
            seen = df[df["step"] <= i]
            stable = seen[seen["loss"].notna()]
            dead = seen[seen["loss"].isna()]
            a = artists[label]

            a["past"].set_offsets(seen[["timestep_fs", "epsilon"]].to_numpy())
            a["crashes"].set_offsets(
                dead[["timestep_fs", "epsilon"]].to_numpy() if len(dead) else empty)

            improving = stable[stable["loss"].cummin() == stable["loss"]]
            if len(improving):
                a["chain"].set_data(improving["timestep_fs"], improving["epsilon"])
                a["chain_pts"].set_offsets(improving[["timestep_fs", "epsilon"]].to_numpy())
                if single:
                    a["chain_pts"].set_array(improving["step"].to_numpy())
                best = improving.iloc[-1]
                a["star"].set_offsets([[best["timestep_fs"], best["epsilon"]]])
                lines.append(f"best loss {best['loss']:.3f}" if single
                             else f"{label}: {best['loss']:.3f}")
            else:
                a["chain"].set_data([], [])
                a["chain_pts"].set_offsets(empty)
                a["star"].set_offsets(empty)
                lines.append("no stable run yet" if single else f"{label}: --")

            current = seen.iloc[-1]
            a["latest"].set_offsets([[current["timestep_fs"], current["epsilon"]]])
            a["curve"].set_data(seen["step"], seen["best_so_far"])
            tail = seen["best_so_far"].to_numpy()
            a["head"].set_offsets([[i, tail[-1]]] if not np.isnan(tail[-1]) else empty)

        readout.set_text("   ".join(lines) if single else "\n".join(lines))
        return []

    fig.tight_layout()

    # Rounded corners, applied after tight_layout because the radii are derived
    # from the final axes geometry.
    # Only the artists drawn in `ax`: clipping the convergence curves to the
    # landscape's path would put them outside their own panel and hide them.
    on_landscape = ("past", "crashes", "chain", "chain_pts", "latest", "star")
    animated = [arm[k] for arm in artists.values() for k in on_landscape]
    round_axes(ax, field + animated)
    round_axes(cbar.ax, [cbar.solids], radius_in=0.045)
    round_corner_elbow(ax2)

    anim = FuncAnimation(fig, update, frames=list(range(1, n_runs + 1)), blit=False)
    _write_animation(anim, OUT / out_name)


def main() -> None:
    # Deliberately the same seed as the blue arm of the comparison GIF below, so
    # the two animations show the same run. Seed 0 is 6th best of the 20 seeds
    # (0.166 against a median of 0.181) — better than typical, but not an outlier.
    build([("GP", "gp_trajectory.csv", BLUE)], "trajectory.gif")
    build([("Random warm-up", "gp_trajectory.csv", BLUE),
           ("LLM warm-up", "llm_trajectory.csv", AQUA)],
          "trajectory-warmstart.gif")
    build([("Random warm-up", "gp_trajectory.csv", BLUE),
           ("LLM warm-up", "llm_trajectory.csv", AQUA),
           ("LLM only", "llm_only_trajectory.csv", VIOLET)],
          "trajectory-three-arms.gif")


if __name__ == "__main__":
    main()

"""Figures for the Bayesian-optimisation-for-MD post.

Run with a python that has matplotlib + pandas (not the optimiser venv):

    python3 app/research/research/_bayesopt-md-scratch/make_plots.py

Reads the CSVs written by generate_data.py and writes SVGs into
public/bayesopt-for-md-simulators/.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from pathlib import Path  # noqa: E402

SCRATCH = Path(__file__).parent
OUT = Path("/Users/federico/Documents/personal/code/website/public/bayesopt-for-md-simulators")
OUT.mkdir(parents=True, exist_ok=True)

# --- house palette (shared with the multi-agent post) ---
SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK_SECONDARY = "#52514e"; INK_MUTED = "#898781"
GRID = "#e1e0d9"; BASE = "#c3c2b7"
BLUE = "#2a78d6"; ORANGE = "#eb6834"; AQUA = "#1baf7a"; YELLOW = "#eda100"
CRITICAL = "#d03b3b"

# Sequential ramp for the loss field: one hue, monotone light -> dark.
# Verified monotone in relative luminance (0.896 -> 0.022).
LOSS_CMAP = LinearSegmentedColormap.from_list(
    "loss", ["#fdf1e7", "#f9d3b4", "#f2ab77", "#eb6834", "#b8451c", "#7d2c10", "#4a1a09"]
)

# Run-order ramp for the search path: one hue, monotone light -> dark
# (relative luminance 0.755 -> 0.031), distinct in hue from the loss ramp.
RUN_CMAP = LinearSegmentedColormap.from_list(
    "run", ["#d3e3f7", "#a8c8ee", "#6ea3e0", "#2a78d6", "#1c548f", "#0f3355"]
)

# Categorical assignment, fixed order, validated for CVD separation.
ARM_COLOR = {"GP (tuned)": BLUE, "GP (untuned)": AQUA, "Random": ORANGE}

LOSS_FLOOR = 0.1465
THRESHOLD = 0.18
# Mean spend per simulation for the tuned GP, measured from trajectories.csv.
USD_PER_SIM = 68.5

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": BASE, "axes.labelcolor": INK_SECONDARY,
    "text.color": INK, "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 11,
    "svg.fonttype": "none",
})


def style_ax(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)
        ax.spines[s].set_linewidth(1)
    ax.grid(False)
    ax.grid(axis=grid_axis)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=10)


def save(fig, name):
    # dpi applies to rasterized layers only (the heatmap meshes); text and lines
    # stay vector, so the file is small and still sharp.
    fig.savefig(OUT / name, format="svg", facecolor=SURFACE, bbox_inches="tight", dpi=200)
    print("wrote", OUT / name)
    plt.close(fig)


# ---------------------------------------------------------------- figure 1 ---
def figure_landscape():
    land = pd.read_csv(SCRATCH / "landscape.csv")
    boundary = pd.read_csv(SCRATCH / "stability_boundary.csv")

    eps_vals = np.sort(land["epsilon"].unique())
    dt_vals = np.sort(land["timestep_fs"].unique())
    grid = land.pivot(index="epsilon", columns="timestep_fs", values="loss").values
    crashed = np.isnan(grid)

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.4))
    vmin, vmax = np.nanmin(grid), 0.55
    contour_levels = [0.18, 0.22, 0.28, 0.36, 0.46]

    for i, ax in enumerate(axes):
        # Crash region: texture + flat neutral, so it never reads as a loss value.
        ax.pcolormesh(dt_vals, eps_vals, np.where(crashed, 1.0, np.nan),
                      cmap=LinearSegmentedColormap.from_list("x", [BASE, BASE]),
                      shading="auto", zorder=1, rasterized=True)
        hatch = ax.contourf(dt_vals, eps_vals, crashed.astype(float), levels=[0.5, 1.5],
                            colors="none", hatches=["////"], zorder=2)
        hatch.set_rasterized(True)  # ContourSet is a single artist in matplotlib >= 3.8
        mesh = ax.pcolormesh(dt_vals, eps_vals, grid, cmap=LOSS_CMAP, shading="auto",
                             vmin=vmin, vmax=vmax, alpha=1.0 if i == 0 else 0.45,
                             zorder=3, rasterized=True)
        if i == 0:
            surface_mesh = mesh
        cs = ax.contour(dt_vals, eps_vals, grid, levels=contour_levels,
                        colors=[INK_SECONDARY if i == 0 else INK_MUTED],
                        linewidths=0.7, alpha=0.55, zorder=4)
        if i == 0:
            ax.clabel(cs, inline=True, fontsize=7.5, fmt="%.2f", colors=INK_SECONDARY)
        ax.plot(boundary["max_stable_timestep_fs"], boundary["epsilon"],
                color=CRITICAL, linewidth=1.8, zorder=6)
        ax.set_xlim(dt_vals.min(), dt_vals.max())
        ax.set_ylim(eps_vals.min(), eps_vals.max())
        ax.set_xlabel("Timestep (fs)")
        style_ax(ax)
        ax.grid(False)

    axes[0].set_ylabel("Lennard-Jones well depth ε (kJ/mol)")

    # No extend arrow: losses above vmax simply saturate at the darkest step.
    cbar = fig.colorbar(surface_mesh, ax=axes, fraction=0.03, pad=0.015)
    cbar.set_label("Loss vs experiment (lower is better)", color=INK_SECONDARY, fontsize=10)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=9, color=INK_MUTED, labelcolor=INK_MUTED)

    # --- panel 1: the landscape itself ---
    ax = axes[0]
    ax.plot([19.16], [3.1006], marker="o", markersize=10, markerfacecolor="none",
            markeredgecolor=SURFACE, markeredgewidth=3.2, zorder=7)
    ax.plot([19.16], [3.1006], marker="o", markersize=10, markerfacecolor="none",
            markeredgecolor=INK, markeredgewidth=1.6, zorder=8)

    # --- panel 2: the GP's search path, in order ---
    ax = axes[1]
    gp = pd.read_csv(SCRATCH / "gp_trajectory.csv")
    gp["loss"] = pd.to_numeric(gp["loss"], errors="coerce")
    # Every proposal, faint: the exploration the acquisition function insists on.
    handles = [ax.scatter(gp["timestep_fs"], gp["epsilon"], s=30, facecolor=INK_MUTED,
                          edgecolor="none", alpha=0.5, zorder=6,
                          label="every GP proposal")]

    # The chain of runs that actually improved on the best so far. Connecting
    # consecutive proposals instead would be unreadable: the acquisition function
    # keeps darting to the corners of the space and back.
    improving = gp[gp["loss"].notna()].copy()
    improving = improving[improving["loss"].cummin() == improving["loss"]]
    ax.plot(improving["timestep_fs"], improving["epsilon"], color=BLUE, linewidth=1.8,
            alpha=0.75, zorder=8, solid_capstyle="round")
    ax.scatter(improving["timestep_fs"], improving["epsilon"], c=improving["step"],
               cmap=RUN_CMAP, s=90, edgecolor=SURFACE, linewidth=1.2, zorder=9,
               vmin=1, vmax=gp["step"].max())
    handles.append(ax.scatter([], [], s=80, facecolor=BLUE, edgecolor=SURFACE,
                              linewidth=1.2, label="runs that improved on the best"))

    dead = gp[gp["stable"] == False]  # noqa: E712
    handles.append(ax.scatter(dead["timestep_fs"], dead["epsilon"], s=105, marker="x",
                              color=CRITICAL, linewidth=1.9, zorder=11,
                              label="crashed, paid for anyway"))

    best = gp.loc[gp["loss"].idxmin()]
    handles.append(ax.scatter([best["timestep_fs"]], [best["epsilon"]], s=280, marker="*",
                              facecolor=YELLOW, edgecolor=INK, linewidth=1.0, zorder=12,
                              label="best run"))

    # The true optimum of this slice, for reference.
    ax.plot([19.16], [3.1006], marker="o", markersize=10, markerfacecolor="none",
            markeredgecolor=SURFACE, markeredgewidth=3.2, zorder=10)
    ax.plot([19.16], [3.1006], marker="o", markersize=10, markerfacecolor="none",
            markeredgecolor=INK, markeredgewidth=1.6, zorder=10)
    handles.append(ax.scatter([], [], s=80, facecolor="none", edgecolor=INK,
                              linewidth=1.6, label="true optimum of this slice"))

    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.06),
               ncol=5, frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    save(fig, "s1-loss-landscape.svg")


# ---------------------------------------------------------------- figure 2 ---
def _median_band(df, arm, x_col, x_grid):
    """Median and IQR of best-so-far for one arm, interpolated onto a common x."""
    curves = []
    for _, seed_df in df[df["arm"] == arm].groupby("seed"):
        seed_df = seed_df.sort_values("step")
        best = seed_df["best_so_far"].ffill().to_numpy(dtype=float)
        x = seed_df[x_col].to_numpy(dtype=float)
        ok = ~np.isnan(best)
        if not ok.any():
            continue
        # Step interpolation: best-so-far only changes when a run improves on it.
        idx = np.searchsorted(x[ok], x_grid, side="right") - 1
        curve = np.where(idx >= 0, best[ok][np.clip(idx, 0, None)], np.nan)
        curves.append(curve)
    stack = np.vstack(curves)
    return (np.nanmedian(stack, axis=0),
            np.nanpercentile(stack, 25, axis=0),
            np.nanpercentile(stack, 75, axis=0))


def figure_convergence():
    traj = pd.read_csv(SCRATCH / "trajectories.csv")
    arms = ["GP (tuned)", "GP (untuned)", "Random"]
    # Labels are staggered because two arms finish on almost the same value.
    label_dy = {"GP (tuned)": 0, "GP (untuned)": 7, "Random": -7}

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0))

    # --- panel 1: per simulation ---
    ax = axes[0]
    steps = np.arange(1, int(traj["step"].max()) + 1)
    for arm in arms:
        med, lo, hi = _median_band(traj, arm, "step", steps)
        ax.fill_between(steps, lo, hi, color=ARM_COLOR[arm], alpha=0.13, linewidth=0, zorder=3)
        ax.plot(steps, med, color=ARM_COLOR[arm], linewidth=2.2, zorder=5)
        ax.annotate(arm, xy=(steps[-1], med[-1]), xytext=(5, label_dy[arm]),
                    textcoords="offset points", fontsize=9.5, color=ARM_COLOR[arm],
                    va="center", fontweight="bold")
    ax.set_xlabel("Simulations run")
    ax.set_ylabel("Best loss so far (median of 20 seeds)")
    ax.set_xlim(1, steps[-1] * 1.02)
    style_ax(ax)
    ax.set_title("What each optimiser buys per simulation", loc="left",
                 fontsize=12, fontweight="bold", color=INK)

    # --- panel 2: per dollar of GPU time ---
    ax = axes[1]
    # Start where every seed has already run at least one simulation, otherwise
    # the median is taken over a changing subset of seeds and wobbles.
    first_spend = traj[traj["step"] == 1].groupby("arm")["cumulative_cost"].max().max()
    spend = np.linspace(first_spend, 1900, 200)
    for arm in arms:
        med, lo, hi = _median_band(traj, arm, "cumulative_cost", spend)
        ax.fill_between(spend, lo, hi, color=ARM_COLOR[arm], alpha=0.13, linewidth=0, zorder=3)
        ax.plot(spend, med, color=ARM_COLOR[arm], linewidth=2.2, zorder=5)
        ax.annotate(arm, xy=(spend[-1], med[-1]), xytext=(5, label_dy[arm]),
                    textcoords="offset points", fontsize=9.5, color=ARM_COLOR[arm],
                    va="center", fontweight="bold")
    ax.set_xlabel("Cumulative GPU spend (USD)")
    ax.set_xlim(first_spend, spend[-1] * 1.02)
    style_ax(ax)
    ax.set_title("...and what it buys per dollar", loc="left", fontsize=12,
                 fontweight="bold", color=INK)

    for ax in axes:
        ax.axhline(THRESHOLD, color=INK_MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
        ax.axhline(LOSS_FLOOR, color=BASE, linewidth=1, linestyle=(0, (2, 2)), zorder=2)
        ax.set_ylim(0.13, 0.62)
    axes[0].annotate("good enough (0.18)", xy=(2, THRESHOLD), xytext=(0, 4),
                     textcoords="offset points", fontsize=8.5, color=INK_MUTED)
    axes[0].annotate("noise floor (0.147)", xy=(2, LOSS_FLOOR), xytext=(0, -12),
                     textcoords="offset points", fontsize=8.5, color=INK_MUTED)

    fig.suptitle("An untuned GP is worth no more than random search",
                 x=0.005, ha="left", fontsize=14, fontweight="bold", color=INK, y=1.06)
    fig.text(0.005, 1.0,
             "Median best loss across 20 fixed seeds; bands are the interquartile range. "
             "All three arms get the same budget of 35 simulations. Lower is better.",
             fontsize=9, color=INK_MUTED)
    fig.tight_layout()
    save(fig, "s2-convergence.svg")


# ---------------------------------------------------------------- figure 3 ---
def figure_sweeps():
    m = pd.read_csv(SCRATCH / "matrix_results.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9))

    # --- panel 1: how failure is reported ---
    ax = axes[0]
    pen = m[m["phase"] == "1_penalty"].sort_values("unstable_penalty")
    x = np.arange(len(pen))
    ax.plot(x, pen["median_best_loss"], color=BLUE, linewidth=2.2, marker="D",
            markersize=7, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}" for v in pen["unstable_penalty"]])
    ax.set_xlabel("Loss reported to the GP for a crashed run")
    ax.set_ylabel("Median best loss", color=INK_SECONDARY)
    ax.axhline(THRESHOLD, color=INK_MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
    ax.set_ylim(0.172, 0.248)
    style_ax(ax)
    # The low end looks fine on loss alone; its cost shows up as wasted runs.
    ax.annotate("too soft and the GP hunts\ncrashes: 36% of runs die,\nagainst 19% at 0.7",
                xy=(0.05, 0.1852), xytext=(0.2, 0.2265), fontsize=8.5, color=INK_SECONDARY,
                arrowprops=dict(arrowstyle="->", color=INK_MUTED, linewidth=0.9))
    ax.annotate("too harsh and it flattens\nthe GP's view of the valley", xy=(6.0, 0.2335),
                xytext=(2.75, 0.2035), fontsize=8.5, color=INK_SECONDARY,
                arrowprops=dict(arrowstyle="->", color=INK_MUTED, linewidth=0.9))
    ax.set_title("Both extremes cost you", loc="left", fontsize=12, fontweight="bold", color=INK)

    # --- panel 2: wasted simulations ---
    ax = axes[1]
    fr = m[m["phase"] == "4_free_rejection"]
    labels = ["Re-ask\n(default)", "Reject for free\n(tell the GP)"]
    waste = [fr[fr["label"] == "free_rejection=False"]["p_thermostat_violations"].iloc[0],
             fr[fr["label"] == "free_rejection=True"]["p_thermostat_violations"].iloc[0]]
    unstable = [fr[fr["label"] == "free_rejection=False"]["p_unstable"].iloc[0],
                fr[fr["label"] == "free_rejection=True"]["p_unstable"].iloc[0]]
    xs = np.arange(2)
    w = 0.34
    ax.bar(xs - w / 2 - 0.01, [u * 100 for u in unstable], width=w, color=ORANGE,
           edgecolor=SURFACE, linewidth=2, label="All crashed runs", zorder=4)
    ax.bar(xs + w / 2 + 0.01, [v * 100 for v in waste], width=w, color=CRITICAL,
           edgecolor=SURFACE, linewidth=2, label="Avoidable: thermostat rule", zorder=4)
    for xi, (u, v) in enumerate(zip(unstable, waste)):
        ax.annotate(f"{u * 100:.1f}%", (xi - w / 2 - 0.01, u * 100), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=9, color=INK_SECONDARY)
        ax.annotate(f"{v * 100:.1f}%", (xi + w / 2 + 0.01, v * 100), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=9, color=INK_SECONDARY)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Share of simulations that crashed (%)")
    ax.set_ylim(0, 24)
    style_ax(ax)
    ax.legend(loc="upper right", frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY)
    ax.set_title("Stop paying to learn what you know", loc="left",
                 fontsize=12, fontweight="bold", color=INK)

    # --- panel 3: how many simulations to buy ---
    ax = axes[2]
    st = m[m["phase"] == "5_steps"].sort_values("n_steps")
    ax.plot(st["n_steps"], st["median_best_loss"], color=BLUE, linewidth=2.2, marker="D",
            markersize=7, zorder=5)
    for n, l in zip(st["n_steps"], st["median_best_loss"]):
        ax.annotate(f"${n * USD_PER_SIM / 1000:.1f}k", (n, l), xytext=(0, 10),
                    textcoords="offset points", ha="center", fontsize=8.5,
                    color=INK_SECONDARY)
    ax.axhline(THRESHOLD, color=INK_MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
    ax.axhline(LOSS_FLOOR, color=BASE, linewidth=1, linestyle=(0, (2, 2)), zorder=2)
    ax.set_xlabel("Simulations bought (labels show GPU spend)")
    ax.set_ylabel("Median best loss", color=INK_SECONDARY)
    ax.set_ylim(0.14, 0.285)
    ax.set_xlim(5, 80)
    style_ax(ax)
    ax.annotate("good enough (0.18)", xy=(7, THRESHOLD), xytext=(0, 5),
                textcoords="offset points", fontsize=8.5, color=INK_MUTED)
    ax.annotate("noise floor (0.147)", xy=(7, LOSS_FLOOR), xytext=(0, 5),
                textcoords="offset points", fontsize=8.5, color=INK_MUTED)
    ax.set_title("Diminishing returns after ~50 runs", loc="left",
                 fontsize=12, fontweight="bold", color=INK)

    fig.suptitle("Three changes to the optimiser, each measured on its own",
                 x=0.005, ha="left", fontsize=14, fontweight="bold", color=INK, y=1.07)
    fig.text(0.005, 1.005,
             f"20 fixed seeds per condition. Each panel varies one setting; everything else is pinned "
             f"to the best config found so far. Spend assumes ${USD_PER_SIM:.0f} per completed simulation.",
             fontsize=9, color=INK_MUTED)
    fig.tight_layout()
    save(fig, "s3-tuning-sweeps.svg")




# ---------------------------------------------------------------- figure 4 ---
def figure_warm_start():
    """Where the LLM-designed warm-up helps, and where it does not."""
    traj = pd.read_csv(SCRATCH / "warmstart_trajectories.csv")
    traj["loss"] = pd.to_numeric(traj["loss"], errors="coerce")
    arms = {"random": ("Random warm-up", BLUE), "llm": ("LLM warm-up", AQUA)}
    n_warmup = 10

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0))

    # --- panel 1: convergence, with the warm-up window marked ---
    ax = axes[0]
    steps = np.arange(1, int(traj["step"].max()) + 1)
    ax.axvspan(1, n_warmup, color=BASE, alpha=0.18, linewidth=0, zorder=1)
    ax.annotate("warm-up:\nthe only thing that differs", xy=(5.5, 0.55), fontsize=8.5,
                color=INK_SECONDARY, ha="center", zorder=6)
    for arm, (label, colour) in arms.items():
        med, lo, hi = _median_band(traj, arm, "step", steps)
        ax.fill_between(steps, lo, hi, color=colour, alpha=0.13, linewidth=0, zorder=3)
        ax.plot(steps, med, color=colour, linewidth=2.2, zorder=5)
        ax.annotate(label, xy=(steps[-1], med[-1]), xytext=(5, 0), textcoords="offset points",
                    fontsize=9.5, color=colour, va="center", fontweight="bold")
    ax.axhline(THRESHOLD, color=INK_MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
    ax.axhline(LOSS_FLOOR, color=BASE, linewidth=1, linestyle=(0, (2, 2)), zorder=2)
    ax.annotate("good enough (0.18)", xy=(12, THRESHOLD), xytext=(0, 4),
                textcoords="offset points", fontsize=8.5, color=INK_MUTED)
    ax.set_xlim(1, steps[-1] * 1.02)
    ax.set_ylim(0.13, 0.62)
    ax.set_xlabel("Simulations run")
    ax.set_ylabel("Best loss so far (median of 20 seeds)")
    style_ax(ax)
    ax.set_title("The gap opens in the warm-up and never closes", loc="left",
                 fontsize=12, fontweight="bold", color=INK)

    # --- panel 2: per-seed outcomes, so the median is not the whole story ---
    ax = axes[1]
    rng = np.random.default_rng(0)  # jitter only, not data
    for i, (arm, (label, colour)) in enumerate(arms.items()):
        final = traj[traj["arm"] == arm].sort_values("step").groupby("seed").tail(1)
        values = pd.to_numeric(final["best_so_far"]).to_numpy()
        x = i + rng.uniform(-0.13, 0.13, size=len(values))
        ax.scatter(x, values, s=54, facecolor=colour, edgecolor=SURFACE, linewidth=1.0,
                   alpha=0.85, zorder=5)
        ax.plot([i - 0.28, i + 0.28], [np.median(values)] * 2, color=INK, linewidth=2.0,
                zorder=6)
        ax.annotate(f"median {np.median(values):.3f}", xy=(i + 0.3, np.median(values)),
                    xytext=(4, 0), textcoords="offset points", ha="left", va="center",
                    fontsize=9, color=INK, fontweight="bold")
        reached = (values <= THRESHOLD).mean()
        ax.annotate(f"{reached:.0%} of seeds reached target", xy=(i, 0.292),
                    fontsize=8.5, color=INK_SECONDARY, ha="center")
    ax.axhline(THRESHOLD, color=INK_MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
    ax.axhline(LOSS_FLOOR, color=BASE, linewidth=1, linestyle=(0, (2, 2)), zorder=2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([arms[a][0] for a in arms])
    ax.set_xlim(-0.5, 1.75)
    ax.set_ylim(0.128, 0.30)
    ax.set_ylabel("Best loss reached (one dot per seed)")
    style_ax(ax)
    ax.set_title("Every seed, not just the median", loc="left", fontsize=12,
                 fontweight="bold", color=INK)

    fig.suptitle("Asking a model for the first ten simulations",
                 x=0.005, ha="left", fontsize=14, fontweight="bold", color=INK, y=1.06)
    fig.text(0.005, 1.0,
             "Both arms are the same tuned optimiser on the same 20 seeds, differing only in where "
             "the first 10 parameter sets come from: uniform-random draws, or a design proposed by "
             "Claude given the system description and bounds.",
             fontsize=9, color=INK_MUTED)
    fig.tight_layout()
    save(fig, "s4-warm-start.svg")


if __name__ == "__main__":
    figure_landscape()
    figure_convergence()
    figure_sweeps()
    figure_warm_start()

"""All four completed arms side by side: the molecule each one is proposing.

One panel per arm, every panel drawn from that arm's own seed-0 trajectory,
over a shared convergence plot. The point is comparative -- the grey reference
molecule is identical in all four, so the panels can be read against each other
and against the answer at the same time.

Arms that finish early (the stall guard, or a run that ends short) hold their
final frame, so the panels stay aligned to a common simulation count rather
than each running at its own speed.

    python3 app/research/research/_bayesopt-md-scratch/make_glycerol_arms.py
"""

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
    AQUA, BASE, BLUE, CRITICAL, INK_MUTED, INK_SECONDARY, ORANGE, SURFACE,
    VIOLET, YELLOW, _write_animation, round_axes, round_corner_elbow, style_ax,
)
from make_molecule_3d import (  # noqa: E402
    AZIMUTH, CORE_FRACTION, FLOOR_Y, LEAN, MASK, ROCK_AMPLITUDE, SHADE,
    _rgba, _rx, _view,
)
from glycerol_runs import all_seeds, representative_seed, steps_csv  # noqa: E402
from make_molecule_animation import (  # noqa: E402
    LOSS_FLOOR, THRESHOLD, TRUTH, geometry, load,
)

OUT = Path("/Users/federico/Documents/personal/code/website/public/bayesopt-for-md-simulators")
REPO = Path("/Users/federico/Documents/personal/code/agentic-optimiser")
DPI = 130

# Three arms, same budget, same information. Both the surrogate and the LLM
# are given the reference targets -- which is the realistic setup, since a
# practitioner fitting a force field has the reference data in hand. The
# surrogate uses them through a multi-output model (one GP per measured
# observable, loss composed analytically) plus the analytically knowable
# feasibility constraints, so it is the strong configuration rather than the
# scalar-loss one.
#
# Random search is the floor, not a competitor.
ARMS = [
    ("random", "Random search", "Random", YELLOW),
    ("gp_multi", "Bayesian optimisation", "BO", BLUE),
    ("llm_only", "LLM", "LLM", ORANGE),
]


def _best_series(rows):
    out, cur = [], float("inf")
    for r in rows:
        if r["best_so_far"] is not None:
            cur = r["best_so_far"]
        out.append(cur)
    return out


def main() -> None:
    data = []
    for tag, label, short, colour in ARMS:
        seed = representative_seed(tag)
        src = [steps_csv(tag, seed)] if steps_csv(tag, seed) else []
        if not src:
            print(f"  skipping {tag}: no run_0")
            continue
        rows = load(src[0])
        data.append({"tag": tag, "label": label, "short": short,
                     "colour": colour, "rows": rows, "best": _best_series(rows)})
    n = max(len(d["rows"]) for d in data)

    fig = plt.figure(figsize=(11.0, 6.0), dpi=DPI)
    fig.patch.set_facecolor(BASE)
    gs = fig.add_gridspec(2, len(data), height_ratios=[1.18, 1.0],
                          hspace=0.24, wspace=0.06,
                          left=0.055, right=0.985, top=0.965, bottom=0.085)
    mol_axes = [fig.add_subplot(gs[0, i]) for i in range(len(data))]
    ax2 = fig.add_subplot(gs[1, :])
    for a in (*mol_axes, ax2):
        a.set_facecolor(SURFACE)

    lim = 0.62
    for ax, d in zip(mol_axes, data):
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim * 0.86, lim * 0.86)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(d["short"], fontsize=8.5, color=INK_SECONDARY, pad=6)

    ax2.set_xlim(1, n)
    top = max(max(b for b in d["best"] if math.isfinite(b)) for d in data)
    # Log scale: the arms span 0.13 to 3.5, and on a linear axis the three
    # that matter are pressed into the bottom eighth of the panel against the
    # noise floor. The interesting differences are all at the low end.
    ax2.set_yscale("log")
    ax2.set_ylim(LOSS_FLOOR * 0.7, top * 1.25)
    ax2.set_xlabel("Simulations run")
    ax2.set_ylabel("Best loss so far")
    style_ax(ax2)
    ax2.axhline(THRESHOLD, color=INK_MUTED, lw=1.0, ls="--", zorder=1)
    ax2.text(1.4, THRESHOLD, "good enough", ha="left", va="bottom",
             fontsize=8, color=INK_MUTED)
    ax2.axhline(LOSS_FLOOR, color=INK_MUTED, lw=0.9, ls=":", zorder=1)
    ax2.text(1.4, LOSS_FLOOR, "noise floor", ha="left", va="bottom",
             fontsize=8, color=INK_MUTED)
    curves = [ax2.plot([], [], color=d["colour"], lw=2.2, zorder=3,
                       label=d["label"])[0] for d in data]
    leg = ax2.legend(loc="upper right", frameon=False, fontsize=8, ncol=2,
                     labelcolor=INK_SECONDARY, handlelength=1.6)
    for t in leg.get_texts():
        t.set_color(INK_SECONDARY)
    round_corner_elbow(ax2)
    round_axes(ax2, curves)
    for ax in mol_axes:
        round_axes(ax, [])

    truth_xy = geometry(TRUTH["bond_length_nm"], TRUTH["angle_deg"])
    truth3 = np.column_stack([truth_xy[:, 0], truth_xy[:, 1], np.zeros(3)])
    truth3 -= truth3.mean(axis=0)
    truth_w = (_rx(LEAN) @ truth3.T).T

    def frame(k):
        rock = ROCK_AMPLITUDE * math.sin(2 * math.pi * k / max(n - 1, 1))
        view = _view(AZIMUTH + rock)

        def to_screen(pts):
            cam = (view @ pts.T).T
            return cam[:, :2], cam[:, 2]

        truth_s, truth_d = to_screen(truth_w)

        for ax, d in zip(mol_axes, data):
            for art in list(ax.images) + list(ax.lines) + list(ax.texts) + list(ax.collections):
                art.remove()
            idx = min(k, len(d["rows"]) - 1)
            r = d["rows"][idx]
            colour = d["colour"] if r["stable"] else CRITICAL

            g = 0.42
            for t in np.linspace(-g, g, 6):
                for seg in (np.array([[-g, FLOOR_Y, t], [g, FLOOR_Y, t]]),
                            np.array([[t, FLOOR_Y, -g], [t, FLOOR_Y, g]])):
                    xy, _ = to_screen(seg)
                    ax.plot(xy[:, 0], xy[:, 1], color=INK_MUTED, lw=0.45,
                            alpha=0.28, zorder=2)

            xy2 = geometry(r["bond_length_nm"], r["angle_deg"])
            cand = np.column_stack([xy2[:, 0], xy2[:, 1], np.zeros(3)])
            cand -= cand.mean(axis=0)
            cand = (_rx(LEAN) @ cand.T).T
            cand_s, cand_d = to_screen(cand)

            for i, c in enumerate(cand):
                fs, _ = to_screen(np.array([[c[0], FLOOR_Y, c[2]]]))
                ax.plot([cand_s[i, 0], fs[0, 0]], [cand_s[i, 1], fs[0, 1]],
                        color=colour, lw=0.6, alpha=0.32, ls=(0, (2, 2)), zorder=3)
                ax.scatter(fs[0, 0], fs[0, 1], s=60, color=INK_MUTED,
                           alpha=0.15, linewidths=0, zorder=3)

            queue = [(dep, c, TRUTH["sigma"] / 2 * CORE_FRACTION, INK_MUTED, 0.5)
                     for c, dep in zip(truth_s, truth_d)]
            alpha = 0.45 + 0.50 * min(r["epsilon"] / 8.0, 1.0)
            for c, dep in zip(cand_s, cand_d):
                queue.append((dep, c, r["sigma"] / 2 * CORE_FRACTION, colour, alpha))
                queue.append((dep - 1e-6, c, r["sigma"] / 2, colour, 0.10))

            for i, j in ((0, 1), (1, 2)):
                ax.plot(*zip(truth_s[i], truth_s[j]), color=INK_MUTED, lw=1.1,
                        ls=(0, (4, 3)), alpha=0.45, solid_capstyle="round",
                        zorder=10 + (truth_d[i] + truth_d[j]) / 2 * 100)
                ax.plot(*zip(cand_s[i], cand_s[j]), color=colour, lw=2.6,
                        solid_capstyle="round",
                        zorder=10 + (cand_d[i] + cand_d[j]) / 2 * 100)

            for dep, c, rad, col, al in sorted(queue, key=lambda q: q[0]):
                ax.imshow(_rgba(col, SHADE, MASK, al),
                          extent=(c[0] - rad, c[0] + rad, c[1] - rad, c[1] + rad),
                          zorder=10 + dep * 100, interpolation="bilinear")

            note = "crashed" if not r["stable"] else f"loss {r['loss']:.3f}"
            ax.text(0.5, 0.025, f"b₀ {r['bond_length_nm']:.3f}\n{note}",
                    transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=7, color=INK_SECONDARY, family="monospace",
                    linespacing=1.4, zorder=200)

        for curve, d in zip(curves, data):
            m = min(k, len(d["best"]) - 1)
            pts = [(x + 1, y) for x, y in enumerate(d["best"][:m + 1])
                   if math.isfinite(y)]
            curve.set_data([q[0] for q in pts], [q[1] for q in pts])
        return []

    anim = FuncAnimation(fig, frame, frames=n, blit=False)
    _write_animation(anim, OUT / "glycerol-arms.gif")
    plt.close(fig)


if __name__ == "__main__":
    main()

"""The box each arm ended up with, drawn against the one experiment implies.

The bead figure can only show sigma as a radius. What it cannot show is the
thing actually being fitted: under NPT the box finds its own volume, so density
*is* the box size. Four arms that disagree about density are four boxes of
visibly different size holding the same 125 molecules.

The dashed cube is the box those 125 molecules would occupy at glycerol's
measured density, 1258.4 kg/m3 -- computable exactly, no simulation needed:

    V = N * M / (rho * N_A)   ->   side = 2.477 nm

Each arm's champion for one seed is re-simulated to recover a *trajectory*; the
optimisers are not re-run, only replayed, so this costs four simulations rather
than a study. Two things move in the animation and both are the physics: the
beads diffuse, and the box breathes as the barostat works against the target
pressure.

Drawing conventions, stated because they are choices:
* Beads are drawn at 0.55 * sigma/2. At full sigma a liquid at this density is
  space-filling and renders as an opaque blob -- true to the physics, useless as
  a picture. Reduced radii keep the spacing legible.
* Depth fades the fill, so the front face reads as the front face.
* Positions are wrapped into the box, as periodic boundaries require, so a bead
  leaving one face reappears on the opposite one. That is real, not a glitch --
  and it is why no bonds are drawn: a molecule straddling a face would otherwise
  be joined by a line across the whole picture.

    MD_ENGINE=glycerol_real python3 make_real_box.py
"""
import csv
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
    AQUA, BASE, BLUE, CRITICAL, INK, INK_MUTED, INK_SECONDARY, ORANGE, VIOLET,
    _write_animation, style_ax,
)
from make_molecule_3d import AZIMUTH, _rx, _view  # noqa: E402

WHITE = "#ffffff"
RUNS = Path("/Users/federico/Documents/personal/code/agentic-optimiser/runs")
OUT = Path("/Users/federico/Documents/personal/code/website/public/bayesopt-for-md-simulators")
SNAPSHOTS = SCRATCH / "_real_target_runs" / "box_snapshots.npz"

SEED = 7
RHO_STAR = 1258.4
BEAD_SCALE = 0.55          # drawn radius as a fraction of sigma/2
# Production is 3000 steps at 20 fs = 60 ps. Every 60th step gives 50 frames,
# over which beads diffuse a few tenths of a nanometre -- visible at this scale.
TRAJECTORY_EVERY = 60
ARM_SPECS = [
    ("Bayesian Optimization (BO)", "glycerol_real", VIOLET),
    ("BO + feasibility model", "glycerol_real_gpfeas", AQUA),
    ("BO + LLM warm-start", "glycerol_real_llmwarm", BLUE),
    ("LLM", "glycerol_real_llm", ORANGE),
]


def reference_box_nm(n_mol: int = 125) -> float:
    """Box side those molecules occupy at the measured density."""
    from optimiser.cgmd import GLYCEROL_MASS_U
    volume = n_mol * GLYCEROL_MASS_U * 1.66053906660 / RHO_STAR
    return volume ** (1.0 / 3.0)


def champion(run_dir: Path, seed: int) -> dict | None:
    d = run_dir / f"run_{seed}"
    cands = sorted(d.glob("*/steps.csv"), key=lambda p: p.stat().st_size)
    if not cands:
        return None
    rows = [r for r in csv.DictReader(cands[-1].open())
            if r["loss"] not in ("", None)]
    if not rows:
        return None
    b = min(rows, key=lambda r: float(r["loss"]))
    return {"epsilon": float(b["epsilon"]), "sigma": float(b["sigma"]),
            "cutoff_nm": float(b["cutoff_nm"]), "loss": float(b["loss"]),
            "density": float(b["density_kg_m3"]),
            "dhvap": float(b["dhvap_kj_mol"]) if b.get("dhvap_kj_mol") else None}


def collect() -> dict:
    """Replay each arm's champion once, keeping the final configuration."""
    from optimiser.cg_real import (BAROSTAT_TAU_PS, COMPRESSIBILITY_PER_BAR,
                                   FROZEN, N_EQUILIBRATE, N_PRODUCTION,
                                   TARGET_PRESSURE_BAR, dhvap_from_potential)
    from optimiser.cgmd import run_glycerol_md

    store = {}
    for name, d, _c in ARM_SPECS:
        champ = champion(RUNS / d, SEED)
        if champ is None:
            print(f"  {name}: no run for seed {SEED}")
            continue
        print(f"  {name}: replaying eps={champ['epsilon']:.2f} "
              f"sigma={champ['sigma']:.3f} cutoff={champ['cutoff_nm']:.3f}",
              flush=True)
        r = run_glycerol_md(
            epsilon=champ["epsilon"], sigma=champ["sigma"],
            cutoff_nm=champ["cutoff_nm"], **FROZEN, seed=SEED,
            target_pressure_bar=TARGET_PRESSURE_BAR,
            barostat_tau_ps=BAROSTAT_TAU_PS,
            compressibility_per_bar=COMPRESSIBILITY_PER_BAR,
            n_equilibrate=N_EQUILIBRATE, n_production=N_PRODUCTION,
            trajectory_every=TRAJECTORY_EVERY)
        if not r["stable"]:
            print(f"    crashed on replay: {r['violations']}")
            continue
        traj = r["trajectory"]
        store[f"{name}|pos"] = traj["positions"]        # (frames, beads, 3)
        store[f"{name}|boxes"] = traj["box_nm"]
        # run_glycerol_md is the raw engine: it reports potential energy, and
        # DHvap is what the scoring layer makes of it.
        o = r["observables"]
        store[f"{name}|meta"] = np.array([
            float(traj["box_nm"][-1]), champ["sigma"], champ["epsilon"],
            champ["cutoff_nm"],
            o["density_kg_m3"], dhvap_from_potential(o["potential_per_molecule"]),
            champ["loss"]])
    SNAPSHOTS.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SNAPSHOTS, **store)
    print(f"wrote {SNAPSHOTS}")
    return store


def cube_edges(side: float) -> list[np.ndarray]:
    c = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)]) * side
    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            if np.count_nonzero(np.abs(c[i] - c[j]) > 1e-9) == 1:
                edges.append(np.stack([c[i], c[j]]))
    return edges


def setup_panel(ax, ref_side, colour, label):
    """Static furniture: the reference cube and the axes limits."""
    ax.clear()
    ax.set_facecolor(WHITE)
    ax.axis("off")
    ax.set_aspect("equal")
    half = ref_side * 0.98
    ax.set_xlim(-half, half)
    ax.set_ylim(-half * 0.95, half * 1.02)
    view = _view(AZIMUTH) @ _rx(math.radians(-14.0))
    for e in cube_edges(ref_side):
        q = (e - np.array([ref_side] * 3) / 2.0) @ view.T
        ax.plot(q[:, 0], q[:, 1], color=CRITICAL, lw=1.2, ls=(0, (4, 3)),
                alpha=0.6, zorder=1)
    ax.set_title(label, fontsize=12, color=colour, loc="center", pad=2)
    return view, half


def points_per_nm(ax, half):
    """Marker sizes are in points squared, so the scale has to be measured."""
    bbox = ax.get_window_extent()
    return bbox.width / (2.0 * half) * 72.0 / ax.figure.dpi


def draw_frame(ax, view, half, pos, box, sigma, colour, scatter, edges):
    """Update one panel: the measured box outline and the beads inside it."""
    centre = np.array([box] * 3) / 2.0
    for line, e in zip(edges, cube_edges(box)):
        q = (e - centre) @ view.T
        line.set_data(q[:, 0], q[:, 1])

    cam = (pos - centre) @ view.T
    order = np.argsort(cam[:, 2])
    cam = cam[order]
    depth = cam[:, 2]
    span = max(depth.max() - depth.min(), 1e-9)
    f = (depth - depth.min()) / span
    rgba = np.tile(np.array(matplotlib.colors.to_rgba(colour)), (len(cam), 1))
    rgba[:, 3] = 0.25 + 0.55 * f
    radius = sigma / 2.0 * BEAD_SCALE
    scatter.set_offsets(cam[:, :2])
    scatter.set_facecolor(rgba)
    scatter.set_sizes(np.full(len(cam), (2.0 * radius * points_per_nm(ax, half)) ** 2))


def main():
    if os.environ.get("MD_ENGINE") != "glycerol_real":
        raise SystemExit("run with MD_ENGINE=glycerol_real")
    if SNAPSHOTS.exists() and "--replay" not in sys.argv:
        store = dict(np.load(SNAPSHOTS))
        print(f"using cached trajectories ({SNAPSHOTS.name}); --replay to redo")
    else:
        store = collect()

    ref = reference_box_nm()
    panels = [(n, c) for n, _d, c in ARM_SPECS if f"{n}|pos" in store]
    n_frames = min(store[f"{n}|pos"].shape[0] for n, _c in panels)
    print(f"{len(panels)} arms, {n_frames} frames")

    fig, axes = plt.subplots(2, 2, figsize=(6.8, 6.9), facecolor=WHITE)
    axes = axes.ravel()
    state = []
    for ax, (name, colour) in zip(axes, panels):
        view, half = setup_panel(ax, ref, colour, name)
        edges = [ax.plot([], [], color=INK_MUTED, lw=1.0, alpha=0.7,
                         zorder=2)[0] for _ in cube_edges(ref)]
        scatter = ax.scatter([], [], s=[], linewidths=0.3, edgecolors="white",
                             zorder=3)
        caption = ax.text(0, -half * 0.88, "", ha="center", va="top",
                          fontsize=10, color=INK_SECONDARY)
        state.append((ax, view, half, name, colour, edges, scatter, caption))
    for ax in axes[len(panels):]:
        ax.axis("off")

    header = fig.text(0.5, 0.985, "", ha="center", va="top", fontsize=10.5,
                      color=INK)

    def frame(k):
        for ax, view, half, name, colour, edges, scatter, caption in state:
            pos = store[f"{name}|pos"][k]
            box = float(store[f"{name}|boxes"][k])
            sigma = float(store[f"{name}|meta"][1])
            draw_frame(ax, view, half, pos, box, sigma, colour, scatter, edges)
            rho = RHO_STAR * (ref / box) ** 3
            caption.set_text(f"box {box:.3f} nm   ({ref:.3f} at experiment)\n"
                             f"$\\rho$ {rho:.0f} kg m$^{{-3}}$   "
                             f"({100 * (rho - RHO_STAR) / RHO_STAR:+.1f}%)")
        header.set_text(f"125 glycerol molecules at 1 bar   ·   seed {SEED}   ·   "
                        f"dashed cube = the box experiment implies")
        return []

    frame(n_frames - 1)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "glycerol-real-box.png", dpi=200, facecolor=WHITE)
    print(f"wrote {OUT / 'glycerol-real-box.png'}")

    anim = FuncAnimation(fig, frame, frames=n_frames, blit=False)
    _write_animation(anim, OUT / "glycerol-real-box.webp", facecolor=WHITE)
    print(f"reference box side {ref:.4f} nm at {RHO_STAR} kg/m3")


if __name__ == "__main__":
    main()

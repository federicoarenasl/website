"""Property space: is the fit converging on the right liquid?

This replaces the molecule figure from the synthetic study. That one worked
because four of the nine free parameters *were* the molecule's geometry, so a
candidate could be drawn as a shape and compared against the true one. Here the
bonded parameters are frozen and there is no true parameter set, so every
candidate would draw the same molecule at a different bead radius -- a picture
of one number.

What replaces it is the question the molecule figure was really asking: is the
model converging on the right *object*? Each simulation is a point in the space
of the two measured properties, density and enthalpy of vaporisation, with the
experimental target and its uncertainty marked. A run that is failing on packing
separates from one failing on cohesion, which a scalar loss hides.

Note on the reconstruction, because it is not measured data everywhere: until
the logger was fixed, glycerol_real runs recorded density but not DHvap, so for
the BO baseline DHvap is recovered from the loss identity

    loss = 0.5*|rho - rho*|/rho*  +  0.5*|H - H*|/H*

which fixes |H - H*| but not its sign. The sign is taken from the nearest point
in the measured landscape sweep, which carries both properties directly. Points
whose sign could not be resolved are drawn hollow.

    python3 make_real_property_space.py
"""
import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SCRATCH = Path(__file__).parent
sys.path.insert(0, str(SCRATCH))
from make_animation import (  # noqa: E402
    AQUA, BASE, BLUE, CRITICAL, INK, INK_MUTED, INK_SECONDARY, ORANGE, SURFACE,
    VIOLET, style_ax,
)

RUNS = Path("/Users/federico/Documents/personal/code/agentic-optimiser/runs/glycerol_real")
LANDSCAPE = SCRATCH / "real_landscape.csv"
OUT = Path("/Users/federico/Documents/personal/code/website/public/bayesopt-for-md-simulators")

RHO_STAR, H_STAR = 1258.4, 91.0
RHO_U, H_U = 0.86, 1.0          # experimental uncertainties on each target


def load_landscape():
    if not LANDSCAPE.exists():
        return []
    out = []
    for r in csv.DictReader(LANDSCAPE.open()):
        if r["loss"] == "" or not r["dhvap_kj_mol"]:
            continue
        out.append((float(r["density_kg_m3"]), float(r["dhvap_kj_mol"]),
                    float(r["loss"])))
    return out


def reconstruct_dhvap(loss, rho, sign_lookup):
    """|H - H*| from the loss identity; sign from the measured landscape."""
    residual = loss - 0.5 * abs(rho - RHO_STAR) / RHO_STAR
    magnitude = max(residual, 0.0) * 2.0 * H_STAR
    sign = sign_lookup(rho)
    if sign is None:
        return H_STAR + magnitude, False
    return H_STAR + sign * magnitude, True


def main():
    land = load_landscape()

    # Sign lookup: among measured landscape points at a similar density, is
    # DHvap above or below target? Density is monotone enough in this slice for
    # that to resolve the branch.
    def sign_lookup(rho):
        if not land:
            return None
        near = sorted(land, key=lambda t: abs(t[0] - rho))[:5]
        votes = [1.0 if h > H_STAR else -1.0 for _, h, _ in near]
        return math.copysign(1.0, sum(votes)) if votes else None

    seeds = {}
    for d in sorted(RUNS.glob("run_*")):
        cands = sorted(d.glob("*/steps.csv"), key=lambda p: p.stat().st_size)
        if not cands:
            continue
        rows = [r for r in csv.DictReader(cands[-1].open())
                if r["loss"] not in ("", None) and r.get("density_kg_m3")]
        if rows:
            seeds[d.name] = rows

    fig, ax = plt.subplots(figsize=(7.2, 5.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    if land:
        rr = [t[0] for t in land]
        hh = [t[1] for t in land]
        ll = [t[2] for t in land]
        sc = ax.scatter(rr, hh, c=ll, s=16, cmap="cividis_r", alpha=0.55,
                        linewidths=0, zorder=1)
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label("loss", fontsize=8, color=INK_SECONDARY)
        cb.ax.tick_params(labelsize=7, colors=INK_MUTED)

    # Experimental target and its uncertainty. The box is deliberately drawn to
    # scale: it is far smaller than the spread of the search, which is the point.
    ax.add_patch(plt.Rectangle((RHO_STAR - RHO_U, H_STAR - H_U),
                               2 * RHO_U, 2 * H_U, facecolor=CRITICAL,
                               edgecolor="none", alpha=0.35, zorder=4))
    ax.plot([RHO_STAR], [H_STAR], marker="*", ms=16, color=CRITICAL,
            markeredgecolor="white", markeredgewidth=0.8, zorder=6,
            label="experiment (glycerol, 298.15 K)")

    colours = [BLUE, AQUA, VIOLET, ORANGE, INK_SECONDARY, BASE]
    for i, (name, rows) in enumerate(sorted(seeds.items())[:6]):
        pts = []
        for r in rows:
            rho = float(r["density_kg_m3"])
            logged = r.get("dhvap_kj_mol")
            if logged not in (None, ""):
                pts.append((rho, float(logged), True))
            else:
                h, resolved = reconstruct_dhvap(float(r["loss"]), rho, sign_lookup)
                pts.append((rho, h, resolved))
        # Only improving moves are joined: joining every proposal draws the
        # exploration, which is mostly noise.
        best, keep = math.inf, []
        for (rho, h, ok), r in zip(pts, rows):
            if float(r["loss"]) < best:
                best = float(r["loss"])
                keep.append((rho, h, ok))
        if len(keep) < 2:
            continue
        c = colours[i % len(colours)]
        ax.plot([p[0] for p in keep], [p[1] for p in keep], "-", color=c,
                lw=1.3, alpha=0.85, zorder=5)
        solid = [p for p in keep if p[2]]
        hollow = [p for p in keep if not p[2]]
        if solid:
            ax.plot([p[0] for p in solid], [p[1] for p in solid], "o", ms=4,
                    color=c, zorder=5)
        if hollow:
            ax.plot([p[0] for p in hollow], [p[1] for p in hollow], "o", ms=4,
                    markerfacecolor="none", markeredgecolor=c, zorder=5)
        ax.plot([keep[-1][0]], [keep[-1][1]], "o", ms=7, color=c,
                markeredgecolor="white", markeredgewidth=0.8, zorder=7)

    ax.axvline(RHO_STAR, color=INK_MUTED, lw=0.7, ls=(0, (3, 4)), zorder=2)
    ax.axhline(H_STAR, color=INK_MUTED, lw=0.7, ls=(0, (3, 4)), zorder=2)
    ax.set_xlabel("density  (kg m$^{-3}$)", fontsize=9, color=INK_SECONDARY)
    ax.set_ylabel("enthalpy of vaporisation  (kJ mol$^{-1}$)", fontsize=9,
                  color=INK_SECONDARY)
    ax.set_title("Where each run ends up in property space", fontsize=11,
                 color=INK, loc="left", pad=10)
    style_ax(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower right",
              labelcolor=INK_SECONDARY)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "glycerol-real-property-space.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    print(f"wrote {path}")
    print(f"  {len(land)} measured landscape points, {len(seeds)} seed trajectories")


if __name__ == "__main__":
    main()

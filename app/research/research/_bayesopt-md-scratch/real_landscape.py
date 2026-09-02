"""The real-target loss landscape over the two parameters that matter.

The synthetic study's landscape was drawn over bonded geometry, because that is
what the nine-parameter fit was searching. Here the bonded parameters are frozen
and only three are free, so the field worth drawing is loss over (epsilon,
sigma) -- cohesion against packing -- at the cutoff the search settled on.

Every point is a real NPT simulation scored against experiment, so a crash is a
hole in the field rather than a large value, and the minimum is wherever the
model comes closest to real glycerol rather than a known answer.

    MD_ENGINE=glycerol_real OMP_NUM_THREADS=1 python real_landscape.py
"""
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, "/Users/federico/Documents/personal/code/agentic-optimiser")

if os.environ.get("MD_ENGINE") != "glycerol_real":
    raise SystemExit("run with MD_ENGINE=glycerol_real")

from optimiser.cg_real import run_md_simulation  # noqa: E402

OUT = Path(__file__).parent / "real_landscape.csv"
# The cutoff of the best point found by the BO baseline. Holding it fixed makes
# this a slice, which has to be said out loud: a candidate elsewhere in the
# field might do better at a different cutoff.
CUTOFF = 1.1159
EPSILONS = [round(2.0 + 0.25 * i, 3) for i in range(21)]    # 2.00 .. 7.00
SIGMAS = [round(0.30 + 0.01 * i, 3) for i in range(16)]     # 0.30 .. 0.45
WORKERS = int(os.environ.get("LANDSCAPE_WORKERS", "10"))


def job(args):
    eps, sig = args
    r = run_md_simulation({"epsilon": eps, "sigma": sig, "cutoff_nm": CUTOFF},
                          seed=0)
    o = r.get("observables") or {}
    return {
        "epsilon": eps, "sigma": sig, "cutoff_nm": CUTOFF,
        "stable": r["stable"],
        "violations": "|".join(r["violations"]),
        "loss": r["loss"] if r["loss"] is not None else "",
        "density_kg_m3": o.get("density_kg_m3", ""),
        "dhvap_kj_mol": o.get("dhvap_kj_mol", ""),
    }


def main():
    grid = [(e, s) for e in EPSILONS for s in SIGMAS]
    print(f"{len(grid)} NPT simulations at cutoff {CUTOFF} nm, "
          f"{WORKERS} workers", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for i, row in enumerate(ex.map(job, grid), start=1):
            rows.append(row)
            if i % 20 == 0:
                print(f"  {i}/{len(grid)}", flush=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if r["loss"] != ""]
    print(f"wrote {OUT}: {len(ok)}/{len(rows)} stable")
    if ok:
        best = min(ok, key=lambda r: float(r["loss"]))
        print(f"field minimum {float(best['loss']):.4f} at "
              f"epsilon={best['epsilon']}, sigma={best['sigma']}")


if __name__ == "__main__":
    main()

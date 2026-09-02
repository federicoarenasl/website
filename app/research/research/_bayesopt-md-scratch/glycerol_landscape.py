"""Loss field over bond length and angle, the two parameters being recovered.

Nine dimensions cannot be drawn, so the slice has to be chosen. This one is
picked because it is where the interesting failure lives: every LLM warm-start
design proposed bond_length = 0.300 by reading the mean-bond observable
literally, while the truth is 0.3126, and no arm ever corrected it. A field
over (bond, angle) puts that error on screen.

The other seven parameters are held at the hidden reference values, so the
minimum of this slice sits exactly on the answer. That also means a trajectory
point drawn here has a loss of its own that differs from the field beneath it,
because its other seven parameters are not the reference ones -- the field is
the landscape the optimiser would have seen had it got everything else right.

Streams to disk per row, so stopping early keeps what finished.

    cd /Users/federico/Documents/personal/code/agentic-optimiser
    MD_ENGINE=glycerol ./.venv/bin/python <this file>
"""
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path("/Users/federico/Documents/personal/code/agentic-optimiser")
sys.path.insert(0, str(REPO))
from optimiser.cg_simulator import REFERENCE_PARAMS, run_md_simulation  # noqa: E402

OUT = Path(__file__).parent
N_BOND, N_ANGLE = 20, 20
BONDS = [0.20 + i * (0.40 - 0.20) / (N_BOND - 1) for i in range(N_BOND)]
ANGLES = [80.0 + i * (175.0 - 80.0) / (N_ANGLE - 1) for i in range(N_ANGLE)]


def one(args):
    bond, angle = args
    p = dict(REFERENCE_PARAMS)
    p["bond_length_nm"], p["angle_deg"] = bond, angle
    r = run_md_simulation(p, seed=0)
    return bond, angle, r["loss"], "|".join(r["violations"])


def _completed(path: Path) -> set:
    """Grid points already in the CSV, so a restart resumes rather than repeats."""
    if not path.exists():
        return set()
    import csv as _csv
    with path.open() as f:
        return {(round(float(r["bond_length_nm"]), 5), round(float(r["angle_deg"]), 3))
                for r in _csv.DictReader(f) if r["bond_length_nm"]}


if __name__ == "__main__":
    path = OUT / "glycerol_landscape.csv"
    done_already = _completed(path)
    grid = [(b, a) for b in BONDS for a in ANGLES
            if (round(b, 5), round(a, 3)) not in done_already]
    if done_already:
        print(f"resuming: {len(done_already)} points already on disk, "
              f"{len(grid)} to go", flush=True)
    print(f"{len(grid)} simulations "
          f"(bond {BONDS[0]:.3f}-{BONDS[-1]:.3f}, angle {ANGLES[0]:.0f}-{ANGLES[-1]:.0f})",
          flush=True)
    started = time.time()
    done = crashed = 0
    mode = "a" if done_already else "w"
    with path.open(mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=["bond_length_nm", "angle_deg", "loss", "violations"])
        if not done_already:
            w.writeheader()
        # Half the cores by default rather than all of them: the full machine
        # pins every core at 100% and the fans become the loudest thing in the
        # room, which is a real cost when the work is a background sweep. Set
        # LANDSCAPE_WORKERS to override.
        workers = int(os.environ.get("LANDSCAPE_WORKERS", "6")) or None
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for bond, angle, loss, v in ex.map(one, grid, chunksize=4):
                w.writerow({"bond_length_nm": round(bond, 5),
                            "angle_deg": round(angle, 3),
                            "loss": round(loss, 6) if loss is not None else "",
                            "violations": v})
                done += 1
                crashed += loss is None
                if done % 40 == 0:
                    f.flush()
                    el = time.time() - started
                    print(f"  {done}/{len(grid)}  {el/60:.1f} min, "
                          f"~{(len(grid)-done)*el/done/60:.1f} left ({crashed} crashed)",
                          flush=True)
    print(f"wrote glycerol_landscape.csv | {done} points, {crashed} crashed")

"""Loss field over any two parameters, with the other seven held at reference.

Generalises glycerol_landscape.py so the three-panel figure can have a
background on every panel rather than only the geometry one. Resumable and
streamed, so an interrupt costs only the points still outstanding.

    cd /Users/federico/Documents/personal/code/agentic-optimiser
    MD_ENGINE=glycerol ./.venv/bin/python <this file> bond_k angle_k
"""
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path("/Users/federico/Documents/personal/code/agentic-optimiser")
sys.path.insert(0, str(REPO))
from optimiser.cg_simulator import PARAM_BOUNDS, REFERENCE_PARAMS, run_md_simulation  # noqa: E402

OUT = Path(__file__).parent
N = 20


def one(args):
    xk, yk, xv, yv = args
    p = dict(REFERENCE_PARAMS)
    p[xk], p[yk] = xv, yv
    r = run_md_simulation(p, seed=0)
    return xv, yv, r["loss"], "|".join(r["violations"])


def completed(path):
    if not path.exists():
        return set()
    with path.open() as f:
        return {(round(float(r["x"]), 6), round(float(r["y"]), 6))
                for r in csv.DictReader(f) if r["x"]}


if __name__ == "__main__":
    xk, yk = sys.argv[1], sys.argv[2]
    xs = [PARAM_BOUNDS[xk][0] + i * (PARAM_BOUNDS[xk][1] - PARAM_BOUNDS[xk][0]) / (N - 1)
          for i in range(N)]
    ys = [PARAM_BOUNDS[yk][0] + i * (PARAM_BOUNDS[yk][1] - PARAM_BOUNDS[yk][0]) / (N - 1)
          for i in range(N)]
    path = OUT / f"glycerol_field_{xk}__{yk}.csv"
    done_already = completed(path)
    grid = [(xk, yk, x, y) for x in xs for y in ys
            if (round(x, 6), round(y, 6)) not in done_already]
    print(f"{xk} x {yk}: {len(grid)} to run "
          f"({len(done_already)} already on disk)", flush=True)
    if not grid:
        raise SystemExit(0)

    workers = int(os.environ.get("LANDSCAPE_WORKERS", "4")) or None
    started, done, crashed = time.time(), 0, 0
    with path.open("a" if done_already else "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["x", "y", "loss", "violations"])
        if not done_already:
            w.writeheader()
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for xv, yv, loss, v in ex.map(one, grid, chunksize=4):
                w.writerow({"x": round(xv, 6), "y": round(yv, 6),
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
    print(f"wrote {path.name} | {done} points, {crashed} crashed")

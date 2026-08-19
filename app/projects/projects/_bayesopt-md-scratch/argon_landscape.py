"""Loss field over (sigma, epsilon) for real argon MD.

Every grid point is an actual ~11-second simulation, so the grid is 20x20 = 400
of them rather than the surrogate's 48,400 — a 40x40 version takes over half an
hour of full-machine compute, which is not worth it for a background field.

Results stream to disk as they complete, so stopping early keeps whatever
finished rather than losing everything.

    cd /Users/federico/Documents/personal/code/agentic-optimiser
    MD_ENGINE=argon ./.venv/bin/python <this file>
"""
import csv, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path("/Users/federico/Documents/personal/code/agentic-optimiser")
sys.path.insert(0, str(REPO))
from optimiser.simulator import run_md_simulation  # noqa: E402

OUT = Path(__file__).parent
N = 20
# Held at settings the optimiser converges on. dt = 25 fs puts the blow-up
# boundary inside this window, so the cliff is visible in the slice.
FIXED = {"timestep_fs": 25.0, "thermostat_tau_ps": 0.5, "cutoff_nm": 1.0}
EPS = [0.5 + i * (2.0 - 0.5) / (N - 1) for i in range(N)]
SIG = [0.30 + i * (0.42 - 0.30) / (N - 1) for i in range(N)]


def one(args):
    eps, sig = args
    r = run_md_simulation({"epsilon": eps, "sigma": sig, **FIXED}, seed=0)
    return eps, sig, (r["loss"] if r["stable"] else None), "|".join(r["violations"])


if __name__ == "__main__":
    grid = [(e, s) for e in EPS for s in SIG]
    print(f"{len(grid)} real simulations at dt={FIXED['timestep_fs']} fs", flush=True)
    done = crashed = 0
    with (OUT / "argon_landscape.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epsilon", "sigma", "loss", "violations"])
        w.writeheader()
        with ProcessPoolExecutor() as ex:
            for e, s, loss, v in ex.map(one, grid, chunksize=4):
                w.writerow({"epsilon": round(e, 5), "sigma": round(s, 5),
                            "loss": round(loss, 6) if loss is not None else "",
                            "violations": v})
                done += 1
                crashed += loss is None
                if done % 40 == 0:
                    f.flush()
                    print(f"  {done}/{len(grid)} ({crashed} crashed)", flush=True)
    print(f"wrote argon_landscape.csv | {done} points, {crashed} crashed")

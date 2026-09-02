"""Seed-averaged loss field over (sigma, epsilon), batched on the Apple GPU.

The CPU version ran one stochastic simulation per grid point, and it showed:
the contours were pocked with islands that look like secondary minima but are
just sampling noise, since the loss is a finite-trajectory estimator with a
~0.037 noise floor. Averaging N_SEEDS independent runs per point shrinks that
by sqrt(N_SEEDS) and leaves the field the contours are meant to represent --
the expected loss -- rather than one draw from it.

Running it on the GPU is what makes that affordable: 8,000 simulations batched
is minutes, where the CPU took 11 minutes for 400. Metal is float32, but
validate_gpu_engine.py and the 400-point paired comparison both put the
disagreement at zero bias (Pearson r = 1.0000) and within the estimator's own
noise, so an average over GPU runs is an unbiased estimate of the same field.

Streams to disk per grid point, so an interrupt keeps what finished.

    cd /Users/federico/Documents/personal/code/agentic-optimiser
    ./.venv/bin/python <this file>
"""
import csv
import sys
import time
from pathlib import Path

REPO = Path("/Users/federico/Documents/personal/code/agentic-optimiser")
sys.path.insert(0, str(REPO))
from optimiser.md_simulator import run_md_simulation_batch  # noqa: E402

OUT = Path(__file__).parent
N = 40
N_SEEDS = 5
# Same held-fixed slice as the CPU landscape, so the two are comparable.
FIXED = {"timestep_fs": 25.0, "thermostat_tau_ps": 0.5, "cutoff_nm": 1.0}
EPS = [0.5 + i * (2.0 - 0.5) / (N - 1) for i in range(N)]
SIG = [0.30 + i * (0.42 - 0.30) / (N - 1) for i in range(N)]

if __name__ == "__main__":
    points = [(e, s) for e in EPS for s in SIG]
    param_sets, seeds = [], []
    for e, s in points:
        for k in range(N_SEEDS):
            param_sets.append({"epsilon": e, "sigma": s, **FIXED})
            seeds.append(k)

    total = len(param_sets)
    print(f"{total} GPU simulations = {len(points)} grid points x {N_SEEDS} seeds", flush=True)
    started = time.time()

    def progress(done, n):
        el = time.time() - started
        rate = done / max(el, 1e-9)
        print(f"  {done}/{n}  {el/60:.1f} min elapsed, "
              f"~{(n - done)/max(rate, 1e-9)/60:.1f} min left", flush=True)

    results = run_md_simulation_batch(param_sets, seeds=seeds, progress=progress)

    with (OUT / "argon_landscape_gpu.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epsilon", "sigma", "loss", "loss_sd",
                                          "n_stable", "crash_frac", "violations"])
        w.writeheader()
        for i, (e, s) in enumerate(points):
            block = results[i * N_SEEDS:(i + 1) * N_SEEDS]
            good = [r["loss"] for r in block if r["stable"]]
            viol = sorted({v for r in block for v in r["violations"]})
            mean = sum(good) / len(good) if good else None
            sd = ((sum((x - mean) ** 2 for x in good) / (len(good) - 1)) ** 0.5
                  if len(good) > 1 else 0.0)
            w.writerow({
                "epsilon": round(e, 5), "sigma": round(s, 5),
                "loss": round(mean, 6) if mean is not None else "",
                "loss_sd": round(sd, 6), "n_stable": len(good),
                "crash_frac": round(1 - len(good) / N_SEEDS, 3),
                "violations": "|".join(viol),
            })
    print(f"wrote argon_landscape_gpu.csv in {(time.time()-started)/60:.1f} min")

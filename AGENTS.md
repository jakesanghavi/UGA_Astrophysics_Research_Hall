# AGENTS.md

Guidance for AI coding agents working in this repository.

## Repository overview

Astrophysics/astrobiology research code (UGA, Dr. Cassandra Hall's group).

- `prod_job/` — **the only code that is actually run in production.** Everything
  else in the repo is scratch work / exploratory analysis and should be treated
  as secondary.
- `N_Body_Simulator/` + `n_body_simulator.py` — standalone N-body simulator
  (pure Python + NumPy/AstroPy).
- `vegetation_modeling/`, `Sapelo2/`, and the various top-level plotting scripts
  are scratch / earlier iterations of the `prod_job` pipeline.

## Environment

The Cloud Agent environment is repo-managed via `.cursor/environment.json`, which
runs `.cursor/install.sh`. That script:

1. Installs the system toolchain required to compile ExoPlaSim's Fortran climate
   model: `gcc`, `g++`, `gfortran`, and OpenMPI.
2. Creates a `.venv/` virtual environment and installs `requirements.txt`.
3. Pre-compiles ExoPlaSim's `pyfft` postprocessor.

Python dependencies live in the virtual environment. Always activate it first:

```bash
source .venv/bin/activate
```

`exoplasim` compiles the PlaSim model the first time a model of a given
resolution/CPU count is run; subsequent runs reuse the compiled binary.

## Running the production job

`prod_job/run_job.sh` is a **SLURM batch script for the Sapelo2 HPC cluster**
(`#SBATCH`, `module load`, `srun`, and a cluster-specific virtualenv path). It
will not run as-is on a local machine or in the Cloud Agent VM — those `module`
and `srun` commands do not exist here.

To run the pipeline locally, invoke `run_model.py` directly with the project
virtualenv:

```bash
source .venv/bin/activate
cd prod_job
python run_model.py
```

(`prod_job/5interpolated_seiss_1E9.dat`, the isochrone `run_model.py` reads at
runtime, is committed, so no copy step is needed.)

### Run modes and the Earth reference

`run_model.py` has a `RUN_MODE` switch at the top:

- `"normal"` — the full sweep over stellar masses (`MSTARS`) and each star's
  habitable-zone distances. Output: `16cpus_test_<mass>.json`.
- `"mass_only"` — vary planet mass only; every planet sits at 1 AU around a
  1 solar-mass star. Output: `16cpus_test_<mass>_massonly.json` (the `_massonly`
  tag distinguishes it from a normal run).

Regardless of `RUN_MODE`, every run first computes an **Earth reference**
(1 Earth-mass planet at 1 AU around a 1 solar-mass star) and stores it in
`earth_reference.json`, giving a known-normal point to compare against.

### Plotting

`plot_veg_by_params.py` (run from `prod_job/` after a normal-mode sweep) reads
the `16cpus_test_<mass>.json` files and writes bar-grid PNGs of normalized GPP
vs planet mass for each (M*, AU) cell. It uses the shared `paper.mplstyle`
style and treats crashed (`null`) points as zero.

## Testing

**The production code is slow.** A full `run_model.py` sweeps every mass ratio in
`MASS_RATIOS` × every star in `MSTARS` × 3 habitable-zone distances, each running
`N_YEARS` sequential ExoPlaSim years. Do **not** run the full grid to smoke-test.

For a quick end-to-end test, shrink the workload rather than editing the
committed source, e.g. from a throwaway script:

```python
import model_helpers as mh
mh.N_YEARS = 1                       # one model year instead of 5
mh.MSTARS = [1.0]                    # one star instead of five
mh.calc_hz_percentiles = lambda m: [mh.calc_hz_percentiles(m)[1]]  # one HZ distance
mh.model_fun(1.0, resolution="T21")  # one mass ratio
```

A single T21 ExoPlaSim year takes on the order of a couple of minutes.

## Performance (prod_job)

The pipeline is sped up in two functionally-equivalent ways:

- Grid-level parallelism: `model_helpers.WORKERS` controls how many independent
  grid points run at once, each in its own worker process (ExoPlaSim uses
  process-wide `os.chdir`, so concurrency must be process-based, not threads).
  Keep `WORKERS * NCPUS <= physical cores` (e.g. `WORKERS=2`, `NCPUS=4` on an
  8-core M1). `WORKERS=1` restores fully sequential behavior. When `WORKERS > 1`,
  runs launch with `mpiexec --bind-to none` so concurrent `mpiexec` invocations
  don't pin to the same cores (otherwise parallel is *slower*, not faster).
- Fewer subprocess/postprocess passes: all `N_YEARS` run in a single subprocess,
  and intermediate years skip pyburn postprocessing (only the final year, which
  is the only one inspected, is postprocessed).

Because `model_fun` uses a process pool, `run_model.py` guards its sweep with
`if __name__ == "__main__":` — required so macOS `spawn` workers don't re-run it.

Note: ExoPlaSim (dynamic vegetation + chaotic climate) is **not deterministic**
run-to-run — repeating an identical run varies the vegetation output by several
percent. Exact bit-reproducibility is not expected; compare results
distributionally, not by exact equality.

### Crash handling

Low-gravity planets (small mass) at high insolation (close to the star) can hit
an intermittent, seed-dependent numerical instability in PlaSim's surface-flux
scheme (`negative z/z0`), which crashes the run. `calculate_veg` retries a
crashed run up to `MAX_RETRIES` (default 2) times with a fresh random seed. If
every attempt still crashes, the grid point's vegetation entries are recorded as
JSON `null` (`[null, null, startemp, flux]`), which is distinct from a genuine
zero-vegetation result (`[0.0, 0.0, ...]`). On a rerun, successfully-computed
points are skipped but `null` (crashed) points are re-attempted.

## Cursor Cloud specific instructions

- Use the `.venv` created by `.cursor/install.sh`; do not `pip install` into the
  system Python (it is externally managed / PEP 668).
- When testing `prod_job`, always reduce the parameter grid as shown above and
  keep model years small — never launch the full multi-hour sweep.

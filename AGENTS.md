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
# run_model.py / veg_utils.calc_hz_percentiles read this isochrone from the CWD:
cp ../Sapelo2/5interpolated_seiss_1E9.dat .
python run_model.py
```

Note: `5interpolated_seiss_1E9.dat` is git-ignored (`**.dat`) and currently only
lives in `Sapelo2/`, so copy it into the working directory before running.

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

## Cursor Cloud specific instructions

- Use the `.venv` created by `.cursor/install.sh`; do not `pip install` into the
  system Python (it is externally managed / PEP 668).
- When testing `prod_job`, always reduce the parameter grid as shown above and
  keep model years small — never launch the full multi-hour sweep.

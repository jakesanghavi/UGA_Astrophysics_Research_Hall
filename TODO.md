# TODO

Planned improvements for the `prod_job` ExoPlaSim pipeline. **None implemented yet** —
this is a backlog of agreed ideas.

## a) Limit postprocessed output to only the consumed variables
- Configure the pyburn postprocessor to emit only what the pipeline reads
  (`veggpp` = code 300, `lsm` = code 172) instead of all ~119 variables.
- Measured on a single T21 year: postprocessing ~7.8s → ~1.8s and ~18 MB → ~0.07 MB
  per pass. Runtime gain is small now that only the final year is postprocessed;
  the real benefit is disk/I/O (~3 GB → ~12 MB over a full ~165-point grid) and
  less contention when many runs write concurrently.
- Tradeoff: other diagnostics would no longer be archived. Must keep `veggpp` +
  `lsm` (exactly what `inspect()` reads).

## b) Configurable random seed for reproducible runs
- ExoPlaSim is nondeterministic by default: PlaSim adds a white-noise initial-
  condition perturbation (`kick=1`) whose RNG is seeded from the system clock
  when the `SEED` namelist parameter is 0; chaotic dynamics then amplify it
  (~6.5% run-to-run spread in vegetation observed).
- Expose a fixed, nonzero `SEED` via the plasim namelist to make individual runs
  reproducible. Note: exact reproducibility also requires holding `NCPUS` fixed
  (MPI floating-point summation order is not associative).

## c) Seed-ensemble runs to get a distribution
- Run each grid point multiple times with different seeds and aggregate
  (mean / median / spread) so we characterize the vegetation *distribution*
  rather than a single noisy realization.
- Depends on (b). Decide on ensemble size per (mass, star, AU); reuse the
  existing `WORKERS` process pool to parallelize ensemble members.

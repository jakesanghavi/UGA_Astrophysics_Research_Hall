# Running the ExoPlaSim / SIMBA sweep

Production code is `prod_job/`. Atmosphere physics and SIMBA limits are in
`ATMOSPHERES.md`.

## Environment

```bash
source .venv/bin/activate
cd prod_job
```

On Sapelo2 the batch script uses `~/env/exoplasim_16cpu` instead. First PlaSim
compile for a given resolution/CPU count is slow; later runs reuse the binary.

## Local vs cluster

Local / Cloud Agent VM:

```bash
python run_model.py
```

Sapelo2: from `prod_job/`, `sbatch run_job.sh`. Do not wrap it in `srun`.
The driver already launches `mpiexec -np NCPUS` per planet and a process pool
for concurrent planets. `srun` would start one full sweep per SLURM task.

`run_job.sh` sets:

```bash
export EXOPLASIM_NCPUS=4
export EXOPLASIM_WORKERS=$(( SLURM_NTASKS / EXOPLASIM_NCPUS ))
```

Keep `WORKERS * NCPUS <=` cores. Local defaults in `model_helpers.py` are
`NCPUS=4`, `WORKERS=2`. Sequential: `export EXOPLASIM_WORKERS=1`.

## Switches in `run_model.py`

Edit the constants at the top. Then `python run_model.py`.

| Switch | Values | Effect |
|---|---|---|
| `RUN_MODE` | `"normal"` / `"mass_only"` | Full `MSTARS` × HZ distances, or one planet at 1 AU around 1 M⊙ |
| `ATMOS_TYPE` | `n2`, `co2`, `mars`, `evolved`, `venus_surface`, `mars_surface` | Composition. Aliases: `earth`→`n2`, `venus`→`co2` |
| `ATMOS_PARAMS` | `None` or a dict of partial pressures | Overlay on the preset, e.g. `{"pCO2": 0.965, "pN2": 0.035}` |
| `PHYSICS_MODE` | `earth` / `mars` / `other` | PlaSim `gascon` / `akap`. `mars` also compiles `p_mars.f90` (calendar, ozone, soil, orbit) |
| `MASS_RATIOS` | `[0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2]` | Planet masses in M⊕. Leave the committed list alone; subset only in a throwaway script |

Production default: `RUN_MODE = "normal"`, `ATMOS_TYPE = "n2"`,
`PHYSICS_MODE = "earth"`, `ATMOS_PARAMS = None`.

`co2` and `mars` are ~1 bar mixes. Use `venus_surface` (92 bar) or
`mars_surface` (6.36 mbar) only when you want those surface pressures.

`physics_mode="mars"` is for actual Mars, not a Mars-composition planet at
1 AU. For Venus- or Mars-like air around a Sun-like star use `other`.

`N_YEARS` and `MSTARS` live in `model_helpers.py` (`N_YEARS = 3`,
`MSTARS = [0.8, 1.0, 1.2]`). Do not cut them in the committed source for a
smoke test.

Every `run_model.py` invocation first writes `earth_reference.json`: 1 M⊕ at
1 AU around 1 M⊙ with `n2` + earth physics, regardless of the sweep's mix.

## Typical specs

Earth-like production sweep (committed defaults):

```
RUN_MODE = "normal"
ATMOS_TYPE = "n2"
PHYSICS_MODE = "earth"
```

Same planets, mass only (1 M⊙, 1 AU):

```
RUN_MODE = "mass_only"
ATMOS_TYPE = "n2"
PHYSICS_MODE = "earth"
```

Venus-like mix at ~1 bar, Earth gravity constants off, `akap` from the mix:

```
RUN_MODE = "normal"          # or "mass_only"
ATMOS_TYPE = "co2"           # alias: "venus"
PHYSICS_MODE = "other"
```

Mars-like mix at ~1 bar, same physics choice:

```
ATMOS_TYPE = "mars"
PHYSICS_MODE = "other"
```

True Venus / Mars surface pressure (not the production comparison):

```
ATMOS_TYPE = "venus_surface"   # 92 bar
# or
ATMOS_TYPE = "mars_surface"    # 6.36 mbar
PHYSICS_MODE = "other"
```

Evolved H/He leftover on an Earth mix:

```
ATMOS_TYPE = "evolved"
PHYSICS_MODE = "earth"
```

`EXOPLASIM_ATMOS` and `EXOPLASIM_PHYSICS` set `model_helpers` defaults when
the caller does not pass `atmos_type` / `physics_mode`. `run_model.py` always
passes its own `ATMOS_TYPE` / `PHYSICS_MODE`, so those env vars do not change
a `run_model.py` sweep.

## Output files

`16cpus_test_<mass><tag>.json` in `prod_job/`. Mass in the name is
`str(mass).replace(".", "")` (`0.1` → `01`, `1.0` → `10`, `1.5` → `15`).

| Mix / physics | `normal` tag | `mass_only` tag |
|---|---|---|
| `n2` + earth | `_normal_n2` | `_massonly2` |
| anything else, earth physics | `_normal_<atmos>` | `_massonly_<atmos>` |
| anything else, other physics | `_normal_<atmos>_<physics>` | `_massonly_<atmos>_<physics>` |

Examples: `16cpus_test_10_normal_n2.json`, `16cpus_test_15_massonly_co2_other.json`.

Existing successful points are skipped. JSON `null` vegetation means a crash;
those points are retried. `0.0` is a real zero-vegetation result.

## Plotting a production sweep

`plot_veg_by_params.py` reads the per-mass JSON files. The filename tag is
`_massonly` or `_normal` plus `EXTENSION`. Set those to match the table above:

| Files | `MASS_ONLY` | `EXTENSION` |
|---|---|---|
| `_massonly2` (`n2` + earth) | `True` | `"2"` |
| `_normal_n2` (`n2` + earth) | `False` | `"_n2"` |
| `_massonly_co2_other` | `True` | `"_co2_other"` |
| `_normal_mars_other` | `False` | `"_mars_other"` |

```bash
python plot_veg_by_params.py
```

## GPP diagnostic (not the production sweep)

Small grid, CLI instead of editing `run_model.py`. Default: 1 year, not
`N_YEARS`.

```bash
python run_gpp_diagnostics.py --dry-run
```

Earth / Venus-like / Mars-like at ~1 bar, masses 1 and 1.5 M⊕, 1 year:

```bash
python run_gpp_diagnostics.py --atmos n2,co2,mars --masses 1,1.5 --years 1 \
    --physics n2=earth,co2=other,mars=other --output gpp_diag.json
python plot_gpp_diagnostics.py --input gpp_diag.json --outdir gpp_diag_plots
python plot_gpp_diagnostics.py --input gpp_diag.json --outdir gpp_diag_plots --only-bars
```

Other useful flags: `--axis mass|au|mstar`, `--hz` (three HZ distances of
`--mstar`), `--years`, `--force` (recompute even if the JSON already has the
point). `--physics` is one mode for every point, or a per-atmos map.

## Tests and smoke runs

```bash
python test_gpp_terms.py
```

Do not launch the full `MASS_RATIOS` × `MSTARS` × HZ × `N_YEARS` grid to
check that the code runs. Shrink in a throwaway script, not in the committed
files:

```python
import model_helpers as mh
mh.N_YEARS = 1
mh.MSTARS = [1.0]
mh.calc_hz_percentiles = lambda m: [mh.calc_hz_percentiles(m)[1]]
mh.model_fun(1.0, resolution="T21")
```

One T21 year is a few minutes. PlaSim + SIMBA is not bit-reproducible; compare
distributions, not exact GPP.

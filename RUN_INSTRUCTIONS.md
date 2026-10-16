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

`RUN_CONFIGS` is a list. Each entry runs a full mass sweep, then the next
entry starts. Default is `venus_surface` then `mars_surface` (overnight:
92 bar Venus mix, then 6.36 mbar Mars mix). Earth reference still runs once
first (`n2`, earth physics).

| Switch | Values | Effect |
|---|---|---|
| `RUN_MODE` | `"normal"` / `"mass_only"` | Default for entries that omit `run_mode`: full `MSTARS` × HZ, or 1 AU / 1 M⊙ |
| `RUN_CONFIGS` | list of atmos names or dicts | One sweep per entry, in order. Required key: `atmos_type` |
| `MASS_RATIOS` | `[0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2]` | Planet masses in M⊕. Leave the committed list alone; subset only in a throwaway script |

Per-config optional keys: `physics_mode`, `atmos_params`, `ozone`, `run_mode`,
`rotationperiod`, `synchronous`, `obliquity`, `extras`. Omitted physics/ozone:
`other` and no ozone for CO2-dominated presets; `earth` and ozone for `n2` /
`evolved`. Earth day is written unless overridden.

Gravity, radius, timestep, and (for `evolved`) leftover H/He still follow
planet mass inside `calculate_veg`. Timestep also shrinks when surface P is
far from 1 bar (about 5 min at 92 bar, 3 min at 6.36 mbar, then halved on
crash retries).

`co2` and `mars` are ~1 bar mixes. Use `venus_surface` (92 bar) or
`mars_surface` (6.36 mbar) only when you want those surface pressures.
`mars_surface` pins the lid to 5% of P; `venus_surface` keeps Earth's 50 mbar
lid with log-ish layers so the photosphere stays in the domain. Dry mixes
turn off evaporation (`drycore`). First extreme-P run compiles an 8-byte
binary. Delete the old `*_crashed/` workdirs (or just rerun — the driver
already removes them per attempt). JSON `null` points are retried.

`physics_mode="mars"` is for actual Mars, not a Mars-composition planet at
1 AU. For Venus- or Mars-like air around a Sun-like star use `other`.

`N_YEARS` and `MSTARS` live in `model_helpers.py` (`N_YEARS = 3`,
`MSTARS = [0.8, 1.0, 1.2]`). Do not cut them in the committed source for a
smoke test.

Every `run_model.py` invocation first writes `earth_reference.json`: 1 M⊕ at
1 AU around 1 M⊙ with `n2` + earth physics, regardless of the sweep's mix.

## Typical specs

Default overnight (committed):

```
RUN_CONFIGS = [
    {"atmos_type": "venus_surface"},
    {"atmos_type": "mars_surface"},
]
```

Earth-like only:

```
RUN_CONFIGS = [{"atmos_type": "n2"}]
```

Mass only (1 M⊙, 1 AU), Venus-like 1 bar then Mars-like 1 bar:

```
RUN_MODE = "mass_only"
RUN_CONFIGS = ["co2", "mars"]
```

Evolved H/He leftover on an Earth mix:

```
RUN_CONFIGS = [{"atmos_type": "evolved"}]
```

`EXOPLASIM_ATMOS` and `EXOPLASIM_PHYSICS` set `model_helpers` defaults when
the caller does not pass `atmos_type` / `physics_mode`. `run_model.py` always
passes each config's mix, so those env vars do not change a `run_model.py`
sweep.

## Output files

`16cpus_test_<mass><tag>.json` in `prod_job/`. Mass in the name is
`str(mass).replace(".", "")` (`0.1` → `01`, `1.0` → `10`, `1.5` → `15`).
T21 has no resolution suffix; T42 is `_T42` before `.json`.

| Mix / physics | `normal` tag | `mass_only` tag |
|---|---|---|
| earth physics | `_normal_<atmos>` | `_massonly_<atmos>` |
| other / mars physics | `_normal_<atmos>_<physics>` | `_massonly_<atmos>_<physics>` |

Examples: `16cpus_test_10_normal_n2.json`, `16cpus_test_10_massonly_n2.json`,
`16cpus_test_15_massonly_co2_other.json`. Older `_massonly2` / `_massonly`
files are still read as `n2` + earth.

Existing successful points are skipped. JSON `null` vegetation means a crash;
those points are retried. `0.0` is a real zero-vegetation result.

## Plotting a production sweep

`plot_veg_by_params.py` discovers every `16cpus_test_*.json` in `--dir`
(default: cwd), groups by (mode, atmos, physics, resolution), and writes
`gpp_<mode>_<atmos>_<physics>[_<res>][_scale].png`. Baseline is
`earth_reference.json`. Masses, M*, and AU come from the files.

```bash
python plot_veg_by_params.py
python plot_veg_by_params.py --list
python plot_veg_by_params.py --atmos n2 --mode mass_only
python plot_veg_by_params.py --atmos co2 --physics other --mode normal
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

`plot_gpp_diagnostics.py` also writes `gpp_rates_by_atmos.png` and
`gpp_rate_regime_bars.png` (light, water, and min) and
`gpp_water_dln_attribution.png` (`P` and plant CO2 vs leftover in `vegppw`).

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

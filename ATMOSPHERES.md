# Atmospheres and SIMBA vegetation

How `prod_job` sets air composition, PlaSim thermodynamics, and the SIMBA
photosynthesis limits.

## Routing

Set these in `run_model.py`:

- `RUN_CONFIGS` — list of sweeps, run in order. Each entry has `atmos_type`
  and may set `physics_mode`, `atmos_params`, `ozone`, `run_mode`.
- `RUN_MODE` — default for entries that omit `run_mode`: `"normal"` (MSTARS ×
  HZ) or `"mass_only"` (1 Msun, 1 AU)
- `MASS_RATIOS` — `[0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2]`. Demos may subset
  this in a throwaway script; leave the committed list alone.

Omitted `physics_mode` is `other` for CO2-dominated presets (`co2`, `mars`,
`venus_surface`, `mars_surface`) and `earth` for `n2` / `evolved`. Omitted
`ozone` is off for those CO2 mixes and on for `n2` / `evolved`. Earth day
(`rotationperiod=1`, `synchronous=False`) is always written unless the entry
overrides it.

Default `RUN_CONFIGS` is four `n2` mass_only sweeps (fixed pressure, then
mass-scaled pressure with each CO2 scaling).

One `model_helpers.py` serves every mix. `model_helpers_n2.py` and
`model_helpers_co2.py` are gone.

Every run writes an Earth reference first: 1 M⊕ at 1 AU around 1 M⊙ with
`n2` + earth physics → `earth_reference.json`.

`EXOPLASIM_ATMOS` and `EXOPLASIM_PHYSICS` override the same defaults from
the environment.

## Composition vs physics

Two independent switches:

| Switch | Where | What it changes |
|---|---|---|
| `ATMOS_TYPE` / `ATMOS_PARAMS` (per `RUN_CONFIGS` entry) | composition | Partial pressures (`pN2`, `pCO2`, …). Whether leftover H/He is written on top. |
| `PHYSICS_MODE` | PlaSim gas constants | How `gascon` (R) and `akap` (κ = R/Cp) are set. |

Venus-like and Mars-like air are both CO2-dominated. They differ in mix (and,
if you opt in, in surface pressure). They do not have to share a physics mode.

`ATMOS_PARAMS` overlays individual partial pressures on a preset so you can
nudge a mix without a second helper file.

### Physics modes

- **earth** (production default): ExoPlaSim derives `gascon` from the partial
  pressures. `akap` stays the compiled Earth/exo value 0.286.
- **mars**: `Model(mars=True)` compiles `p_mars.f90`. That also changes
  calendar, ozone, soil moisture, and orbit defaults. Use for actual Mars, not
  for a Mars-composition planet at 1 AU around a Sun-like star.
- **other**: write `gascon = 8314.46 / mmw` and `akap = R/Cp` from the mix.
  This is the mode for Venus- or Mars-composition planets that should not use
  `p_mars.f90`.

ExoPlaSim already updates `gascon` from the mix. It does **not** update
`akap`; `p_exo.f90` hardcodes 0.286 unless we write it.

### Presets

`co2` / `venus` and `mars` are **mole-fraction mixes at ~1 bar**. That keeps
Venus vs Mars a composition comparison, not a 92 bar vs 6 mbar leap.

| Key | Mix | Surface P | H/He evolution |
|---|---|---|---|
| `n2` (`earth`) | Earth N2/O2 | ~1 bar | no |
| `co2` (`venus`) | Venus mole fractions | ~1 bar | no |
| `mars` | Curiosity/Mahaffy dry air | ~1 bar | no |
| `evolved` | Earth mix, then leftover H/He | ~1 bar + H/He | yes |
| `venus_surface` | same Venus fractions | **92 bar** | no |
| `mars_surface` | same Mars fractions | **6.36 mbar** | no |

`venus` still means `co2` (~1 bar). `mars` still means the 1-bar Mars mix.
`venus_surface` and `mars_surface` are opt-in; they are not production
defaults and are not aliases of `venus` / `mars`.

`venus_surface` / `mars_surface` sweeps use `other` physics and ozone off.
Earth reference is still `n2` + earth physics.

### Output file tags

So two regimes do not overwrite the same JSON:

- `_normal_<atmos>` or `_massonly_<atmos>` when physics is earth
- `_normal_<atmos>_<physics>` / `_massonly_<atmos>_<physics>` otherwise
- plus `_pfix|_pmass` and `_co2mix|_co2pp|_co2area` unless the scaling is
  fixed pressure with `mixing_ratio` CO2

`plot_veg_by_params.py` still reads the old `_massonly2` / `_massonly`
names as `n2` + earth. Non-T21 files get `_Txx` before `.json`.

## Surface pressure

Surface pressure is the sum of the partial pressures (bar). ExoPlaSim
writes that sum to `PSURF`. With `pressure_scaling="fixed"` (default) every
planet mass gets the preset's pressure (1.01 bar for `n2` / `evolved`).
`PRESSURE_FRACTION` in `model_helpers.py` scales all partial pressures by
one global factor; it is 1.

Fixed pressure means the amount of air changes with mass. Column mass is
P/g; total atmosphere is P·4πR²/g.

| Mass (M⊕) | g (m/s²) | column vs Earth | total vs Earth | atm / planet mass vs Earth |
|---|---|---|---|---|
| 0.1 | 3.52 | 2.8× | 0.77× | 7.8× |
| 0.5 | 7.11 | 1.4× | 0.95× | 1.9× |
| 1.0 | 9.62 | 1.0× | 1.04× | 1.04× |
| 2.0 | 13.01 | 0.75× | 1.14× | 0.57× |

`pressure_scaling="mass"` keeps the atmosphere a fixed fraction of planet
mass (Kopparapu et al. 2014 scale background N2 the same way). Then
P = f·G·M²/(4πR⁴); `atmosphere_scale_factors` multiplies every partial
pressure by M²·(R₁/R)⁴, normalized to the code's own 1 M⊕ planet (Noack
radius 1.010 R⊕), so 1 M⊕ keeps the preset. With R ∝ M^0.282 this is
P ≈ 1.01 bar × M^0.872.

`co2_scaling` then sets CO2 on top of that (preset pCO2 = 400 µbar):

| Mass (M⊕) | P (bar) | `mixing_ratio` pCO2 / ppmv | `partial_pressure` | `surface_area` |
|---|---|---|---|---|
| 0.1 | 0.136 | 54 µbar / 396 | 400 µbar / 2942 | 147 µbar / 1080 |
| 0.25 | 0.302 | 119 / 396 | 400 / 1325 | 219 / 725 |
| 0.5 | 0.552 | 219 / 396 | 400 / 725 | 296 / 536 |
| 1.0 | 1.010 | 400 / 396 | 400 / 396 | 400 / 396 |
| 2.0 | 1.849 | 732 / 396 | 400 / 216 | 541 / 293 |

`surface_area` assumes CO2 mass ∝ R², so pCO2 = M_CO2·g/(4πR²) ∝ g
(≈ M^0.436). These are alternative CO2 inventory assumptions, not
predictions of a carbon-silicate cycle.

SIMBA reads CO2 as a mixing ratio. The light-limited β uses plant ppmv
(clamped at 1000, β capped at 1); the water-limited term goes as
surface pressure × plant ppmv, i.e. CO2 partial pressure up to the clamp.
`mixing_ratio` therefore leaves β unchanged across masses;
`partial_pressure` hits the 1000 ppm clamp below ~0.35 M⊕.

Mass-scaled Earth air stays above 0.1 bar down to 0.1 M⊕, so every mass
uses the default vertical grid (`COLUMN_NEAR_EARTH_BAR = (0.1, 2.0)`).

## H/He envelope (`evolved`)

As of 2026-10-30 the envelope has no effect on the climate. Every mass
gets 1e-13 to 1e-12 bar of H2 + He, so `evolved` runs are effectively `n2`
runs. The causes:

- Core-powered loss in `atm_mass_frac.evolve_atmosphere` reduces to
  `eta * M_atm / t_KH`, and `t_KH` depends only on M_atm/M_c. Every mass
  ends at M_atm/M_c = 1.94e-3 at 4.5 Gyr (81% of the initial 1% lost).
- `photoevaporation_rate_paper` uses the Picogna et al. (2019) wind
  profile, which describes X-ray winds from protoplanetary disks, times the
  planet cross-section. It removes ~1e-16 of the envelope. Planetary escape
  is usually energy-limited, `eps * pi * R_p**3 * F_XUV / (G * M_p)`.
- The Bondi limit (`t_cool >= t_Bondi` stops core-powered loss) is
  commented out. With it on, nothing is lost at 1 AU and every planet keeps
  M_atm/M_c = 0.01.
- `core_radius_noack` in `atm_mass_frac.py` takes kg but is passed Earth
  masses, so R_c is ~1 m. R_c cancels in the loss rate, so results are
  unaffected.
- `calculate_veg` divides the core mass by `mass_ratio` before
  `calc_f_ret_big_rcb` (eq. 28, apparently Ginzburg et al. 2016), so f_ret
  uses a core of CMF·M⊕ for every planet: 1.8e-11 at all masses. With the
  core scaled to the planet it runs from 3e-17 (0.1 M⊕) to 1.7e-7
  (2 M⊕). `vegetation_modeling/h_he_retention.py` normalized f_ret to its
  value at 2 M⊕; production does not.
- f_ret multiplies the evolved M_atm/M_c. Both are envelope mass fractions.
  f_ret describes the envelope left after disk dispersal and belongs in
  the initial condition, not as a factor on the end state.
- The pressure `G * M_p * M_env / (4 * pi * R_p**4)` is in Pa and is
  multiplied by `1e-10`. Pa→bar is `1e-5`.

With correct units and no f_ret factor, 1.94e-3 of the core mass gives
~70 bar (0.1 M⊕) to ~1000 bar (2 M⊕) of H/He. Those are sub-Neptunes that
ExoPlaSim cannot run. The planned rebuild uses f_ret (or a cited
post-dispersal fraction) as the starting envelope, energy-limited XUV
escape, a decision on the Bondi limit, and correct units. Masses that keep
an envelope are classed as not Earth-like; masses that lose it get an
Earth-like secondary atmosphere and a GPP run.

## SIMBA

SIMBA (`simba.f90` `vegstep`) is an Earth vegetation scheme. It will run on
other mixes and surface pressures. It is not calibrated for them.

Light-limited GPP:

```
GPP_light = 3.4e-10 * β * f(T) * fPAR * SW_down
GPP       = min(GPP_light, GPP_water)
```

### Plant CO2 vs radiation CO2

PlaSim radiation uses the air mixing ratio `co2 = pCO2 / P_total` (ppmv).
Unpatched SIMBA used that same `co2` in Harvey β and in the water-limited
term (`zgppw ∝ 0.3 * co2`).

At Venus or Mars composition (~95% CO2, ~1 bar) that is ~10^5–10^6 ppm.
Harvey β is `1 + 0.3 ln(C / 360)` with no saturation, so β ≈ 3.5 and
GPP_light scales by that factor even if climate is Earth-like. That is a
mixing-ratio identity, not a fertilization measurement.

The patch splits the two:

- Radiation still uses radmod `co2` (the real air mixing ratio).
- Photosynthesis uses `zco2p = min(air CO2, 1000 ppm)` unless `CO2VEG` is
  set explicitly in the namelist (`CO2VEG >= 0` overrides the clamp).

1000 ppm is a plant-CO2 clamp for the water-limited term (`zgppw ∝ zco2p`).
Harvey β is then capped at 1.0 (the 360 ppm reference). The Earth mix
(400 µbar, 396 ppmv) has β = 1.03 before the cap, so 1.0. High-CO2 mixes cannot get a fertilization bonus above
Earth; low CO2 can still drop β below 1 and to 0 near 13 ppm.

Namelist (written after `configure`): `CO2VEG`, `CO2VEG_MAX`, `T_HOT`,
`T_KILL`. Fortran defaults: `CO2VEG = -1` (use the clamp),
`CO2VEG_MAX = 1000`, `T_HOT = 35`, `T_KILL = 45`. β cap 1.0 is in
`vegstep`, not the namelist.

### Temperature

Original `f(T)` is a thaw switch: 0 below 0 °C, linear to 1 at 5 °C, then
flat. It is not “GPP rises with temperature.”

The patch keeps that 0–5 °C ramp and adds a high-T decline: still 1 at
35 °C, linear to 0 at 45 °C. Not a hard kill at 40 °C.

### Fortran patch

`prod_job/plasim_patches/simba.f90` is copied into the ExoPlaSim package
`plasim/src/simba.f90` on the first `calculate_veg` call if the files
differ. PlaSim is then recompiled (`Model(recompile=True)`). A sha256 stamp
next to the package avoids rebuilding every run.

`co2_sens`, `co2_ref`, and `ct_crit` are Fortran `parameter`s. Changing
those still requires editing the patch and recompiling.

## GPP diagnostic

`run_gpp_diagnostics.py` is a small grid, not the production sweep.
`--physics` is one mode for every point, or a per-atmos map (`n2=earth,co2=other`).

Earth / Venus-like / Mars-like comparison (1-bar mixes, 1 year):

```
python run_gpp_diagnostics.py --atmos n2,co2,mars --masses 1,1.5 --years 1 \
    --physics n2=earth,co2=other,mars=other
```

JSON goes to `gpp_diag.json`; PNGs to `gpp_diag_plots/` (`--no-plot` skips
PNGs). Replot with `python plot_gpp_diagnostics.py`.

It records land-mean β, f(T), fPAR, SW↓, GPP_light (`vegppl`), GPP_water
(`vegppw`), and GPP (`veggpp` = instantaneous min of the two). Plots:

- existing light-factor panels (live GPP only)
- `gpp_rates_by_atmos.png` / `gpp_rate_regime_bars.png`: light, water, and
  min on the same axes per mix (crashed points omitted)
- `gpp_dln_attribution.png`: Δln of the light factors vs an n2 reference
- `gpp_water_dln_attribution.png`: Δln(`vegppw`) split into prescribed `P`
  and plant CO2 plus a climate leftover (ET / VPD / cover / averaging).
  That leftover is not a reconstruction of PlaSim eq. 6.11 from time means.
  Crashed / preset-only rows still show Δln P and Δln plant CO2; leftover
  is blank. `*` on the label.
- `gpp_preset_knobs.png`: P, pCO2, air CO2, plant CO2, β, timestep P-factor,
  modeltop, mmw. Earth (`n2`) is a green X. `plot_gpp_diagnostics.py`
  attaches Earth plus `venus_surface` / `mars_surface` stubs on the same
  (mass, M*, AU) grid when those mixes are missing, so 1 bar / 92 bar /
  6.36 mbar knobs plot without launching a GCM year. `--no-surface-stubs`
  skips the two surface-P stubs; Earth is still attached.

`--preset-only` on `run_gpp_diagnostics.py` writes the same mix/column
fields with no ExoPlaSim, then the PNGs. Default `--atmos` is still
`evolved,n2,co2`.

β in that output is the **plant** (clamped) value. `co2_ppmv` is the air
mixing ratio; `co2veg_ppmv` is what SIMBA used.

Annual-mean of a product is not the product of annual means, so a
reconstructed `GPP_light` from time-averaged factors will not match
`vegppl` exactly.

## Limits

- SIMBA is still an Earth scheme. Off-Earth mixes and surface pressures are
  runnable, not calibrated.
- Production `co2` / `mars` stay ~1 bar. Use `venus_surface` / `mars_surface`
  for 92 bar / 6.36 mbar.
- Surface-P columns get extra configure knobs; ~1 bar mixes are unchanged.
  `mars_surface` pins `modeltop` to 5% of P (`vtype=4`) so the lid is above
  the 6.36 mbar surface. Earth orography stays: ExoPlaSim 3.4.2's
  `orography=` writes `OROSCALE` into `landmod_namelist`, which PlaSim does
  not declare, so the run dies on startup. `venus_surface` keeps the lid at
  Earth's 50 mbar with `vtype=3` (pseudolog) so the ~1 bar CO2 photosphere
  stays in the domain; a 5% top would sit at 4.6 bar and cut that off. Dry
  mixes (`pH2O=0`) set `drycore`. Extreme columns use 8-byte precision, turn
  off pressure broadening, and add `physicsfilter="gp|exp|sp"`.
- Thick columns (> 10 bar) follow the LMD Venus GCM (Lebonnois et al. 2010,
  2016): 20 layers instead of 10, a 1.5 min step, and a VIRA-like starting
  profile (`TGR=735 K`, `ALR=7.8 K/km`, `DTROP=60 km`) instead of PlaSim's
  288 K Earth profile. `ALR` only sets that reference profile. The LMD model
  also makes CO2 heat capacity depend on temperature; PlaSim keeps a fixed
  `akap`, which is not patched here. Thin columns start at 5 min.
- Thick columns also set `DZTOP=2` m (top soil layer, default 0.2 m).
  PlaSim steps land surface temperature explicitly in the atmospheric heat
  flux; at 92 bar the sensible heat flux reached ~1.8 MW/m2 against a 0.2 m
  layer, overshot the air temperature, oscillated to negative kelvin, and
  crashed in `mkstress` (`fluxmod.f90`) after ~2,800 steps. With 2 m the run
  is stable. The ground and ocean still start at Earth temperatures, so the
  column needs years to adjust.
- PlaSim + SIMBA is not bit-reproducible run to run. Compare distributions,
  not exact GPP.

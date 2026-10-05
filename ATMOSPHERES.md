# Atmospheres and SIMBA vegetation

How `prod_job` sets air composition, PlaSim thermodynamics, and the SIMBA
photosynthesis limits.

## Routing

Set these in `run_model.py`:

- `ATMOS_TYPE`, `ATMOS_PARAMS` — composition
- `PHYSICS_MODE` — gascon / akap
- `RUN_MODE` — `"normal"` (MSTARS × HZ) or `"mass_only"` (1 Msun, 1 AU)
- `MASS_RATIOS` — `[0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2]`. Demos may subset
  this in a throwaway script; leave the committed list alone.

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
| `ATMOS_TYPE` / `ATMOS_PARAMS` | composition | Partial pressures (`pN2`, `pCO2`, …). Whether leftover H/He is written on top. |
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

Production default in `run_model.py`: `ATMOS_TYPE = "n2"`, `PHYSICS_MODE = "earth"`.

### Output file tags

So two regimes do not overwrite the same JSON:

- `n2` + earth physics keeps the old names `_normal_n2` / `_massonly2`
- anything else: `_normal_<atmos>` or `_normal_<atmos>_<physics>`
  (and `_massonly_…` in mass-only mode)

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

1000 ppm is a saturation-style cap, not a hard β ceiling. At 1000 ppm,
β ≈ 1.31. Earth at 330 ppm is unchanged (below the clamp). Water-limited
GPP uses `zco2p` as well.

Namelist (written after `configure`): `CO2VEG`, `CO2VEG_MAX`, `T_HOT`,
`T_KILL`. Fortran defaults: `CO2VEG = -1` (use the clamp),
`CO2VEG_MAX = 1000`, `T_HOT = 35`, `T_KILL = 45`.

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

`run_gpp_diagnostics.py` is a small grid, not the production sweep. Example:

```
python run_gpp_diagnostics.py --atmos n2,co2,mars --masses 1,1.5 --years 1 \
    --physics n2=earth,co2=other,mars=other
```

`--physics` can be one mode for every point, or a per-atmos map as above.

It records land-mean β, f(T), fPAR, SW↓, GPP_light, GPP_water, and `Δln` of
those vs an Earth-like reference.

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
- PlaSim + SIMBA is not bit-reproducible run to run. Compare distributions,
  not exact GPP.

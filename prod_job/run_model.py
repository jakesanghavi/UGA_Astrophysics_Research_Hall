from model_helpers import model_fun
from atmos_presets import resolve_atmos_type, resolve_physics_mode, regime_file_tag
from time import time, strftime, gmtime

# --- Run configuration -------------------------------------------------------
# "normal"    : full sweep over stellar masses (MSTARS in model_helpers) and
#               each star's habitable-zone distances.
# "mass_only" : vary planet mass only; every planet is placed at 1 AU around a
#               1 solar-mass star. Output files get a "_massonly" tag so they are
#               easy to distinguish from a normal run.
# RUN_MODE = "mass_only"
RUN_MODE = "normal"

# Composition and physics. See ATMOSPHERES.md.
#   n2 / earth, co2 / venus (~1 bar Venus mix), mars (~1 bar Mars mix),
#   evolved, venus_surface (92 bar), mars_surface (6.36 mbar)
# ATMOS_PARAMS overlays individual partial pressures on the preset.
ATMOS_TYPE = "n2"
ATMOS_PARAMS = None
# ATMOS_PARAMS = {"pCO2": 0.965, "pN2": 0.035}

# earth: ExoPlaSim gascon, akap 0.286
# mars:  p_mars.f90 (calendar/ozone/soil/orbit too)
# other: gascon and akap from the mix
PHYSICS_MODE = "earth"

# Fixed star/orbit used for the Earth reference and for "mass_only" runs.
REFERENCE_MSTAR = 1.0   # solar masses
REFERENCE_AU = 1.0      # AU

MASS_RATIOS = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2]
# MASS_RATIOS = [0.5, 1]


def run_earth_reference():
    """Always-run baseline: a 1 Earth-mass planet at 1 AU around a 1 solar-mass
    star with Earth-like air, stored on its own so every run has a 'normal'
    point to compare to."""
    print("Running Earth reference (1 Mearth, 1 AU, 1 Msun, n2 / earth physics)...")
    model_fun(1.0, resolution="T21",
              points=[(REFERENCE_MSTAR, REFERENCE_AU)],
              output_file="earth_reference.json",
              atmos_type="n2",
              physics_mode="earth")


def main():
    atmos_type = resolve_atmos_type(ATMOS_TYPE)
    physics_mode = resolve_physics_mode(PHYSICS_MODE)
    file_tag = regime_file_tag(RUN_MODE, atmos_type, physics_mode)
    print(f"ATMOS_TYPE={atmos_type}  PHYSICS_MODE={physics_mode}  "
          f"ATMOS_PARAMS={ATMOS_PARAMS}  file_tag={file_tag}")

    # (A) Compute the Earth reference baseline first, regardless of RUN_MODE.
    run_earth_reference()

    for m in MASS_RATIOS:
        print(f"Starting model for mass ratio: {m}")

        start_time = time()

        if RUN_MODE == "mass_only":
            # (B) One planet of mass m at 1 AU around a 1 Msun star.
            model_fun(m, resolution="T21",
                      points=[(REFERENCE_MSTAR, REFERENCE_AU)],
                      file_tag=file_tag,
                      atmos_type=atmos_type,
                      physics_mode=physics_mode,
                      atmos_params=ATMOS_PARAMS)
        else:
            model_fun(m, resolution="T21", file_tag=file_tag,
                      atmos_type=atmos_type,
                      physics_mode=physics_mode,
                      atmos_params=ATMOS_PARAMS)

        end_time = time()
        elapsed_seconds = end_time - start_time

        formatted_time = strftime("%H:%M:%S", gmtime(elapsed_seconds))

        print(f"Finished mass ratio {m}. Execution time: {formatted_time}")
        print("-" * 30)


# The __main__ guard is required: model_fun uses a process pool, and on macOS
# (spawn start method) each worker re-imports this module. Without the guard,
# every worker would relaunch the entire sweep.
if __name__ == "__main__":
    main()

from model_helpers import model_fun
from time import time, strftime, gmtime

# --- Run configuration -------------------------------------------------------
# "normal"    : full sweep over stellar masses (MSTARS in model_helpers) and
#               each star's habitable-zone distances.
# "mass_only" : vary planet mass only; every planet is placed at 1 AU around a
#               1 solar-mass star. Output files get a "_massonly" tag so they are
#               easy to distinguish from a normal run.
RUN_MODE = "mass_only"

# Fixed star/orbit used for the Earth reference and for "mass_only" runs.
REFERENCE_MSTAR = 1.0   # solar masses
REFERENCE_AU = 1.0      # AU

MASS_RATIOS = [0.0266, 0.052, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2, 3, 4]
# MASS_RATIOS = [0.5, 1]


def run_earth_reference():
    """Always-run baseline: a 1 Earth-mass planet at 1 AU around a 1 solar-mass
    star, stored on its own so every run has a 'normal' point to compare to."""
    print("Running Earth reference (1 Mearth, 1 AU, 1 Msun)...")
    model_fun(1.0, resolution="T21",
              points=[(REFERENCE_MSTAR, REFERENCE_AU)],
              output_file="earth_reference.json")


def main():
    # (A) Compute the Earth reference baseline first, regardless of RUN_MODE.
    run_earth_reference()

    for m in MASS_RATIOS:
        print(f"Starting model for mass ratio: {m}")

        start_time = time()

        if RUN_MODE == "mass_only":
            # (B) One planet of mass m at 1 AU around a 1 Msun star.
            model_fun(m, resolution="T21",
                      points=[(REFERENCE_MSTAR, REFERENCE_AU)],
                      file_tag="_massonly")
        else:
            model_fun(m, resolution="T21")

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

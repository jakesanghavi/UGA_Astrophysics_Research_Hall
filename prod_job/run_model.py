from model_helpers import model_fun
from atmos_presets import resolve_run_configs, regime_file_tag
from time import time, strftime, gmtime

# --- Run configuration -------------------------------------------------------
# "normal"    : full sweep over stellar masses (MSTARS in model_helpers) and
#               each star's habitable-zone distances.
# "mass_only" : vary planet mass only; every planet is placed at 1 AU around a
#               1 solar-mass star. Output files get a "_massonly_<atmos>" tag.
# Per-config override: {"atmos_type": "n2", "run_mode": "mass_only"}
RUN_MODE = "normal"

# Sequence of atmosphere sweeps. Each entry is an atmos name or a dict:
#   atmos_type (required)
#   run_mode, physics_mode, atmos_params, ozone  (optional)
#   rotationperiod, synchronous, obliquity, extras, ...  (optional configure())
# Omitted physics_mode / ozone: other + no ozone for CO2 mixes; earth + ozone
# for n2 / evolved. Earth day (rotationperiod=1, synchronous=False) is always
# set unless the entry overrides it.
# Examples:
#   RUN_CONFIGS = [{"atmos_type": "n2"}]
#   RUN_CONFIGS = ["co2", "mars"]
#   RUN_CONFIGS = [{"atmos_type": "n2", "run_mode": "mass_only"}]
RUN_CONFIGS = [
    {"atmos_type": "venus_surface"},
    {"atmos_type": "mars_surface"},
]

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
              physics_mode="earth",
              configure_kwargs={"ozone": True, "rotationperiod": 1.0,
                                "synchronous": False})


def run_one_config(config):
    atmos_type = config["atmos_type"]
    physics_mode = config["physics_mode"]
    run_mode = config["run_mode"]
    file_tag = regime_file_tag(run_mode, atmos_type, physics_mode)
    print(f"ATMOS_TYPE={atmos_type}  PHYSICS_MODE={physics_mode}  "
          f"ATMOS_PARAMS={config['atmos_params']}  ozone={config['ozone']}  "
          f"run_mode={run_mode}  file_tag={file_tag}")

    for m in MASS_RATIOS:
        print(f"Starting model for mass ratio: {m}")

        start_time = time()

        common = dict(
            resolution="T21",
            file_tag=file_tag,
            atmos_type=atmos_type,
            physics_mode=physics_mode,
            atmos_params=config["atmos_params"],
            configure_kwargs=config["extras"],
        )
        if run_mode == "mass_only":
            model_fun(m, points=[(REFERENCE_MSTAR, REFERENCE_AU)], **common)
        else:
            model_fun(m, **common)

        end_time = time()
        elapsed_seconds = end_time - start_time
        formatted_time = strftime("%H:%M:%S", gmtime(elapsed_seconds))
        print(f"Finished mass ratio {m}. Execution time: {formatted_time}")
        print("-" * 30)


def main():
    configs = resolve_run_configs(RUN_CONFIGS, default_run_mode=RUN_MODE)
    run_earth_reference()
    for i, config in enumerate(configs, 1):
        print(f"=== config {i}/{len(configs)}: {config['atmos_type']} ===")
        run_one_config(config)


# The __main__ guard is required: model_fun uses a process pool, and on macOS
# (spawn start method) each worker re-imports this module. Without the guard,
# every worker would relaunch the entire sweep.
if __name__ == "__main__":
    main()

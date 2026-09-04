from model_helpers import model_fun
from time import time, strftime, gmtime

MASS_RATIOS = [0.0266, 0.052, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2, 3, 4]
# MASS_RATIOS = [0.5, 1]


def main():
    for m in MASS_RATIOS:
        print(f"Starting model for mass ratio: {m}")

        start_time = time()

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

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Configuration -----------------------------------------------------------
RESOLUTION = "T21"
# Which value from each JSON entry [avg_gpp, tot_gpp, startemp, flux] to plot.
# Index 1 is total GPP, which is what the Earth baseline is defined on.
VALUE_INDEX = 1

# True: plot a "mass_only" run (every planet at 1 AU / 1 Msun; files tagged
# "_massonly") -- a single row. False: a normal full M*/AU sweep (grid of rows).
MASS_ONLY = True

# Planet masses (Earth masses). File name suffixes are derived from these to
# match run_model.py's naming: str(mass).replace('.', '').
MASS_RATIOS = [0.0266, 0.052, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2, 3, 4]

# Stellar masses shown as subplot rows; only those present in the data are used.
STAR_ROWS = ["0.7", "0.8", "0.9", "1.0", "1.1"]

# Earth reference (1 Mearth, 1 AU, 1 Msun) used to normalize GPP so Earth = 1.
EARTH_REFERENCE_FILE = "earth_reference.json"

# Use the paper style that ships next to this script.
_STYLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper.mplstyle")
plt.style.use(_STYLE)


def mass_suffix(mass):
    """Filename suffix for a planet mass, matching run_model.py output naming."""
    return str(mass).replace(".", "")


def data_tag():
    """Filename tag (resolution + mass_only), matching run_model.py output."""
    res_tag = RESOLUTION if RESOLUTION == "T42" else ""
    return res_tag + ("_massonly" if MASS_ONLY else "")


def load_data():
    """Load {mass_suffix: {mstar: {au: [...]}}} from the per-mass JSON files."""
    tag = data_tag()
    data = {}
    for mass in MASS_RATIOS:
        suffix = mass_suffix(mass)
        filename = f"16cpus_test_{suffix}{tag}.json"
        if not os.path.exists(filename):
            print(f"Warning: {filename} not found; skipping mass {mass}.")
            data[suffix] = {}
            continue
        with open(filename) as f:
            data[suffix] = json.load(f)
    return data


def load_earth_baseline():
    """Earth reference value at VALUE_INDEX (total GPP), used to set Earth = 1.

    Falls back to 1.0 (no normalization) if the reference file is missing.
    """
    if not os.path.exists(EARTH_REFERENCE_FILE):
        print(f"Warning: {EARTH_REFERENCE_FILE} not found; GPP will not be "
              f"normalized to Earth (using 1.0).")
        return 1.0
    with open(EARTH_REFERENCE_FILE) as f:
        ref = json.load(f)
    # earth_reference.json holds a single (M*, AU) point.
    for by_au in ref.values():
        for entry in by_au.values():
            base = entry[VALUE_INDEX]
            if base:  # non-null and nonzero
                return base
    print(f"Warning: no usable baseline in {EARTH_REFERENCE_FILE}; using 1.0.")
    return 1.0


def normalized_gpp(data, suffix, mstar, au, norm):
    """GPP for one (mass, M*, AU), normalized to Earth. 0 if missing or crashed."""
    try:
        val = data[suffix][mstar][au][VALUE_INDEX]
    except (KeyError, IndexError):
        return 0.0
    # Crashed runs are recorded as null (see model_helpers.calculate_veg).
    if val is None:
        return 0.0
    return val / norm


def rows_present(data, star_rows):
    """Star rows that actually have data (keeps a mass_only plot to one row)."""
    suffixes = [mass_suffix(m) for m in MASS_RATIOS]
    present = [s for s in star_rows if any(s in data.get(suf, {}) for suf in suffixes)]
    return present or star_rows


def au_columns(data, star_rows):
    """Sorted AU keys for each star, taken from the first mass file that has them."""
    suffixes = [mass_suffix(m) for m in MASS_RATIOS]
    per_star = {}
    for mstar in star_rows:
        aus = []
        for suffix in suffixes:
            if mstar in data.get(suffix, {}):
                aus = sorted(data[suffix][mstar].keys(), key=float)
                break
        per_star[mstar] = aus
    return per_star


def plot_grid(data, star_rows, norm, yscale="linear", sharey=False, title_suffix=""):
    suffixes = [mass_suffix(m) for m in MASS_RATIOS]
    star_rows = rows_present(data, star_rows)
    per_star = au_columns(data, star_rows)
    ncols = max((len(aus) for aus in per_star.values()), default=1)
    nrows = len(star_rows)

    # Size the figure to the grid so a single-row (mass_only) plot isn't stretched.
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(max(4.0, 4.5 * ncols), max(3.0, 2.8 * nrows)),
                             sharex=True, sharey=sharey, squeeze=False)

    x = np.arange(len(MASS_RATIOS))
    all_values = []
    global_max = -np.inf

    for i, mstar in enumerate(star_rows):
        au_keys = per_star[mstar]
        for j in range(ncols):
            ax = axes[i, j]
            if j >= len(au_keys):
                ax.axis("off")
                continue

            au = au_keys[j]
            values = np.array([normalized_gpp(data, s, mstar, au, norm) for s in suffixes])
            values = np.clip(values, 1e-20, None)
            all_values.extend(values)
            global_max = max(global_max, values.max())

            ax.bar(x, values)
            ax.axhline(1.0, color="0.4", lw=0.8, ls="--")  # Earth reference = 1
            ax.set_title(f"M*={mstar}, AU={round(float(au), 2)}")

            # x labels on the bottom row (the only row in mass_only mode).
            if i == nrows - 1:
                ax.set_xticks(x)
                ax.set_xticklabels(MASS_RATIOS, rotation=90)
                ax.set_xlabel(r"Planet mass [$M_\oplus$]")
            if j == 0:
                ax.set_ylabel("GPP / Earth")
            if yscale == "log":
                ax.set_yscale("log")

    if sharey:
        all_values = np.array(all_values)
        threshold = 1e-15
        valid = all_values[all_values > threshold]
        ymin = valid.min() if valid.size else threshold
        ymax = global_max if global_max > threshold else threshold
        for row in axes:
            for ax in row:
                ax.set_ylim(ymin, ymax)

    fig.suptitle(f"Gross Primary Production (Earth = 1)\n{title_suffix}")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = (title_suffix.replace(" ", "_").replace("(", "")
           .replace(")", "").replace(",", "") + ".png")
    plt.savefig(out)
    plt.close(fig)  # free memory when plotting many grids
    print(f"Wrote {out}")


def main():
    data = load_data()
    norm = load_earth_baseline()
    if MASS_ONLY:
        # A single row -- "shared vs independent" y-scaling is meaningless, so
        # just produce a linear and a log version.
        plot_grid(data, STAR_ROWS, norm, yscale="linear", title_suffix="mass_only linear scale")
        plot_grid(data, STAR_ROWS, norm, yscale="log", title_suffix="mass_only log10 scale")
    else:
        plot_grid(data, STAR_ROWS, norm, yscale="linear", sharey=True,
                  title_suffix="Shared linear scale")
        plot_grid(data, STAR_ROWS, norm, yscale="log", sharey=True,
                  title_suffix="log10 scale (shared)")
        plot_grid(data, STAR_ROWS, norm, yscale="linear", sharey=False,
                  title_suffix="Independent linear scales")


if __name__ == "__main__":
    main()

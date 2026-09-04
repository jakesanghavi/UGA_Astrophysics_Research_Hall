import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Configuration -----------------------------------------------------------
RESOLUTION = "T21"
# Which value from each JSON entry [avg_gpp, tot_gpp, startemp, flux] to plot.
VALUE_INDEX = 0

# Planet masses (Earth masses). File name suffixes are derived from these to
# match run_model.py's naming: str(mass).replace('.', '').
MASS_RATIOS = [0.0266, 0.052, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2, 3, 4]

# Stellar masses shown as subplot rows; each row shows that star's HZ distances.
STAR_ROWS = ["0.7", "0.8", "0.9", "1.0", "1.1"]

# GPP is divided by this reference (an Earth-like GPP) so bars are relative.
# Roughly the value now also stored in earth_reference.json.
EARTH_GPP_NORM = 5.637932898316933e-10

# Use the paper style that ships next to this script.
_STYLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper.mplstyle")
plt.style.use(_STYLE)


def mass_suffix(mass):
    """Filename suffix for a planet mass, matching run_model.py output naming."""
    return str(mass).replace(".", "")


def load_data():
    """Load {mass_suffix: {mstar: {au: [...]}}} from the per-mass JSON files."""
    res_tag = RESOLUTION if RESOLUTION == "T42" else ""
    data = {}
    for mass in MASS_RATIOS:
        suffix = mass_suffix(mass)
        filename = f"16cpus_test_{suffix}{res_tag}.json"
        if not os.path.exists(filename):
            print(f"Warning: {filename} not found; skipping mass {mass}.")
            data[suffix] = {}
            continue
        with open(filename) as f:
            data[suffix] = json.load(f)
    return data


def normalized_gpp(data, suffix, mstar, au):
    """Normalized GPP for one (mass, M*, AU). Returns 0 if missing or crashed."""
    try:
        val = data[suffix][mstar][au][VALUE_INDEX]
    except (KeyError, IndexError):
        return 0.0
    # Crashed runs are recorded as null (see model_helpers.calculate_veg).
    if val is None:
        return 0.0
    return val / EARTH_GPP_NORM


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


def plot_grid(data, star_rows, yscale="linear", sharey=False, title_suffix=""):
    suffixes = [mass_suffix(m) for m in MASS_RATIOS]
    per_star = au_columns(data, star_rows)
    ncols = max((len(aus) for aus in per_star.values()), default=1)

    fig, axes = plt.subplots(len(star_rows), ncols, figsize=(15, 10),
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
            values = np.array([normalized_gpp(data, s, mstar, au) for s in suffixes])
            values = np.clip(values, 1e-20, None)
            all_values.extend(values)
            global_max = max(global_max, values.max())

            ax.bar(x, values)
            ax.set_title(f"M*={mstar}, AU={round(float(au), 2)}")

            if i == len(star_rows) - 1:
                ax.set_xticks(x)
                ax.set_xticklabels(MASS_RATIOS, rotation=90)
            if j == 0:
                ax.set_ylabel("Normalized GPP")
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
        fig.suptitle(f"Gross Primary Production by M*, AU, and Mp.\n{title_suffix}")

    plt.tight_layout()
    out = (title_suffix.replace(" ", "_").replace("(", "")
           .replace(")", "").replace(",", "") + ".png")
    plt.savefig(out)
    plt.close(fig)  # free memory when plotting many grids
    print(f"Wrote {out}")


def main():
    data = load_data()
    plot_grid(data, STAR_ROWS, yscale="linear", sharey=True,
              title_suffix="Shared linear scale")
    plot_grid(data, STAR_ROWS, yscale="log", sharey=True,
              title_suffix="log10 scale (shared)")
    plot_grid(data, STAR_ROWS, yscale="linear", sharey=False,
              title_suffix="Independent linear scales")


if __name__ == "__main__":
    main()

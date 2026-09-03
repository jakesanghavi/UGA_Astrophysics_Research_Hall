import json
import os
import matplotlib.pyplot as plt
import numpy as np

mp_values = ["00266", "0052", "01", "025", "05", "075", "10", "15", "2", "3", "4"]
mp_numeric = [float(m) if (not m.startswith("0") and m != "15" and m!= "10") else float(m[0] + "." + m[1:]) for m in mp_values]
RESOLUTION = "T21"
to_append = RESOLUTION if RESOLUTION == "T42" else ""

data_files = {mp: f"16cpus_test_{mp}{to_append}.json" for mp in mp_values}

# choose which index from the JSON list to plot (0 or 1)
VALUE_INDEX = 0

# ---- LOAD DATA ----
all_data = {}

for mp, filename in data_files.items():
    with open(filename, "r") as f:
        all_data[mp] = json.load(f)

# ---- EXTRACT UNIQUE AXES ----
# outer keys = M*
# inner keys = AU

mstar_keys = sorted(all_data[mp_values[0]].keys(), key=float)

# assume same AU keys structure across files
au_keys_per_mstar = {
    m: sorted(all_data[mp_values[0]][m].keys(), key=float)
    for m in mstar_keys
}


# group1 = ["0.1", "0.5", "1"]
# group2 = ["1.25", "1.5", "2"]

group1 = ["0.7", "0.8", "0.9", "1.0", "1.1"]

# group1 = [0.7, 0.8, 0.9, 1.1]

def plot_grid(mstar_subset, yscale="linear", sharey=False, title_suffix=""):
    fig, axes = plt.subplots(len(mstar_subset), 3,
                             figsize=(15, 10),
                             sharex=True,
                             sharey=sharey)

    # collect all values if we need global y scaling
    all_plot_values = []
    global_min, global_max = np.inf, -np.inf

    for i, mstar in enumerate(mstar_subset):
        au_keys = au_keys_per_mstar[mstar]

        for j, au in enumerate(au_keys):
            ax = axes[i, j]
            ax.tick_params(axis='x', which='major', labelsize=8)

            values = []
            for mp in mp_values:
                try:
                    val = all_data[mp][mstar][au][VALUE_INDEX]
                    val /= 5.637932898316933e-10
                    values.append(val)
                except KeyError:
                    values.append(0)

            values = np.array(values)
            
            values = np.clip(values, 1e-20, None)

            # track global min/max
            all_plot_values.extend(values)
            global_min = min(global_min, values.min())
            global_max = max(global_max, values.max())

            x = np.arange(len(mp_values))
            ax.bar(x, values)

            ax.set_title(f"M*={mstar}, AU={np.round(float(au), 2)}")

            if i == len(mstar_subset) - 1:
                ax.set_xticks(x)
                ax.set_xticklabels(mp_numeric)

            if j == 0:
                ax.set_ylabel("Normalized GPP")

            if yscale == "log":
                ax.set_yscale("log")

    # apply global y-limits if shared
    threshold = 1e-15
    all_plot_values = np.array(all_plot_values)
    valid_mask = all_plot_values > threshold

    if sharey:
        # Clean the limits: if it's smaller than threshold, make it 0 (or the threshold)
        # adjusted_min = global_min if global_min > threshold else 0
        if np.any(valid_mask):
            adjusted_min = all_plot_values[valid_mask].min()
        else:
            adjusted_min = threshold
        adjusted_max = global_max if global_max > threshold else threshold

        for ax_row in axes:
            for ax in ax_row:
                ax.set_ylim(adjusted_min, adjusted_max)

        fig.suptitle(
            f"Gross Primary Production by M*, AU, and Mp.\n{title_suffix}"
        )

    plt.tight_layout()
    # plt.show()
    
    filename = f"{title_suffix.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')}.png"
    plt.savefig(filename, dpi=150)
    plt.close(fig) # Required to free up memory when plotting many grids
    
plot_grid(group1, yscale="linear", sharey=True,
          title_suffix="Shared linear scale")

plot_grid(group1, yscale="log", sharey=True,
          title_suffix="log10 scale (shared)")

plot_grid(group1, yscale="linear", sharey=False,
          title_suffix="Independent linear scales")

group2 = None
if group2 is not None:
    plot_grid(group2, yscale="linear", sharey=False,
            title_suffix="Group 2, independent linear scales")

    plot_grid(group2, yscale="log", sharey=True,
            title_suffix="Group 2, log10 scale (shared)")

    plot_grid(group2, yscale="linear", sharey=True,
            title_suffix="Group 2 (1.25–2.0 M☉), shared linear scale")
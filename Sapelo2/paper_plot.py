import json
import os
import matplotlib.pyplot as plt
import numpy as np

mp_values = ["01", "025", "05", "075", "1", "15", "2", "3", "4"]  # corresponds to 0.5, 1.0, 1.5
mp_numeric = [float(m) if (not m.startswith("0") and m != "15") else float(m[0] + "." + m[1:]) for m in mp_values]
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

# ---- CREATE 3x3 GRID ----
fig, axes = plt.subplots(len(mstar_keys), 3,
                         figsize=(15, 10), sharex=True)

for i, mstar in enumerate(mstar_keys):
    au_keys = au_keys_per_mstar[mstar]
    for j, au in enumerate(au_keys):
        ax = axes[i, j]

        values = []
        for mp in mp_values:
            val = all_data[mp][mstar][au][VALUE_INDEX]
            val /= 5.637932898316933e-10
            values.append(val)

        x = np.arange(len(mp_values))
        ax.bar(x, values)

        ax.set_title(f"M*={mstar}, AU={np.round(float(au), 2)}")

        if i == len(mstar_keys) - 1:
            ax.set_xticks(x)
            ax.set_xticklabels(mp_numeric)

        if j == 0:
            ax.set_ylabel("Normalized GPP")

# ---- GLOBAL LABELS ----
fig.suptitle("Gross Primary Production by M*, AU, and Mp.\nBars normalized to Earth-like case (M*=1, AU=1.13, Mp=1.0).")

plt.tight_layout()
plt.show()
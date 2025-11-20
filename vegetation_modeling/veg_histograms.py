import json
import os
import math
import re
import matplotlib.pyplot as plt
import numpy as np

directory = "."  # adjust as needed


outer_keys = ["0.1", "0.4", "0.7", "1.0", "1.2"]
inner_keys = ["0.1", "0.25", "0.5", "1.0", "1.5", "2.0"]

# Regex to capture MASS_RATIO after MP_
pattern = r"MP_([0-9.]+)\.json$"

results = []

for filename in os.listdir(directory):
    if filename.startswith("wave") and filename.endswith(".json"):
        
        match = re.search(pattern, filename)
        if not match:
            continue

        mass_ratio = str(match.group(1))

        with open(os.path.join(directory, filename), "r") as f:
            data = json.load(f)

        results.append({
            "filename": mass_ratio,
            "data": data
        })

for x in range(len(results)):
    veg_data = {str(mr): {str(au): vals for au, vals in inner.items()} 
                for mr, inner in results[x]['data'].items()}

    # Get rid of non-floats
    new_veg_data = {
        # Outer Key Normalization:
        str(float(outer_k)) if math.floor(float(outer_k)) == math.ceil(float(outer_k)) else outer_k: 
        {
            # Inner Key Normalization:
            str(float(inner_k)) if math.floor(float(inner_k)) == math.ceil(float(inner_k)) else inner_k: inner_v
            for inner_k, inner_v in inner_dict.items()
        }
        for outer_k, inner_dict in veg_data.items()
    }

    results[x]['data'] = new_veg_data

def extract_first_vals(data_list, outer_key, inner_key):
    result = {}

    for entry in data_list:
        filename = entry["filename"]
        data = entry["data"]

        if outer_key in data and inner_key in data[outer_key]:
            vals = data[outer_key][inner_key]
            if isinstance(vals, list) and vals:
                result[filename] = vals[0]

    return result

combos = [(ok, ik) for ok in outer_keys for ik in inner_keys]
n = len(combos)

plots_per_fig = 9
num_figs = int(np.ceil(n / plots_per_fig))

print(f"Total combos: {n}, creating {num_figs} figures.")

for fig_i in range(num_figs):
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    axes = axes.flatten()

    start = fig_i * plots_per_fig
    end = min(start + plots_per_fig, n)
    block = combos[start:end]

    for ax, (outer_key, inner_key) in zip(axes, block):
        extracted = extract_first_vals(results, outer_key, inner_key)

        if not extracted:
            ax.set_title(rf"$M_\oplus$={outer_key}, $au$={inner_key}\n(no data)")
            ax.axis("off")
            continue

        numeric_dict = {float(k): v for k, v in extracted.items()}
        sorted_items = sorted(numeric_dict.items())
        x_vals, y_vals = zip(*sorted_items)

        categories = [str(x) for x in x_vals]

        ax.bar(
            categories,
            y_vals,
            color="skyblue",
            edgecolor="black"
        )

        # LOG-SCALE Y
        ax.set_yscale("log")

        ax.set_title(rf"$M_\oplus$={outer_key}, $au$={inner_key}", fontsize=10)
        ax.tick_params(axis='x', rotation=45)

    # Turn off unused axes
    for ax in axes[len(block):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
import json
import numpy as np
import matplotlib.pyplot as plt

F_INIT = 0.15
MASS_STEP_SIZE = 1
AU_STEP_SIZE = 0.25

filename = f"veg_json_FI_{F_INIT}_MS_{MASS_STEP_SIZE}_AUS_{AU_STEP_SIZE}2.json"

# Load your JSON
with open(filename, "r") as f:
    data = json.load(f)
    
data_float = {float(mr): {float(au): vals for au, vals in inner.items()} 
              for mr, inner in data.items()}

# Extract sorted axes
mass_ratios = sorted(data_float.keys())
aus = sorted({au for inner in data_float.values() for au in inner.keys()})

# Initialize matrix
veg_matrix = np.full((len(mass_ratios), len(aus)), np.nan)

for i, mr in enumerate(mass_ratios):
    for j, au in enumerate(aus):
        if au in data_float[mr]:
            veg_matrix[i, j] = data_float[mr][au][1]
                        
# Plot heatmap
plt.figure(figsize=(8,6))
im = plt.imshow(veg_matrix, origin='lower', 
                extent=[min(aus), max(aus), min(mass_ratios), max(mass_ratios)],
                aspect='auto', cmap='viridis', vmin=0)

plt.colorbar(im, label=f"Total Vegetation (kg C)")
plt.xlabel("Semimajor Axis (AU)")
plt.ylabel("Planetary Mass (Earth Masses)")
plt.title(f"Total Vegetation Heatmap - Initial Gas Fraction = {F_INIT}")
plt.show()

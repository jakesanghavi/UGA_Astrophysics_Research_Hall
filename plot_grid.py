import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec

scale_factor = 10

images = [f'vegetation_map_custom_earthlike_scaled_{scale_factor}_year_{x}.png' for x in range(1, 6)]

fig = plt.figure(figsize=(10, 6))
gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1])

# --- top row: 3 images ---
for i in range(3):
    ax = fig.add_subplot(gs[0, i])
    img = mpimg.imread(images[i])
    ax.imshow(img)
    ax.axis("off")

# --- bottom row: 2 images centered (cols 0–2 → use cols 0:3 with offsets) ---
bottom_positions = [gs[1, 0:2], gs[1, 1:3]]  # shift them right one
for i, pos in enumerate(bottom_positions):
    ax = fig.add_subplot(pos)
    img = mpimg.imread(images[3 + i])
    ax.imshow(img)
    ax.axis("off")

plt.suptitle(f"Earthlike Planet Vegetation by Year - $M={scale_factor}M_\\oplus$")
plt.tight_layout()
plt.show()

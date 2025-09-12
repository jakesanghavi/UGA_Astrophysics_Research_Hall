import exoplasim as exo
from astropy.constants import L_sun, R_sun, sigma_sb
from astropy import units as u
from utils import calculate_intensity_latlon, calculate_P_curve
import numpy as np
from matplotlib import pyplot as plt

# Configure and run the exoplasim model for Earth
earth = exo.Model(workdir="earth",modelname="earth",resolution="T21",ncpus=4,layers=10,precision=8)
earth.configure()
earth.exportcfg()
earth.run(years=1,clean=False)

# Configure parameters for the star of interest (the Sun here)
L_solar = L_sun.to(u.W)
R_solar = R_sun.to(u.m)
T_solar = ((L_sun / (4 * np.pi * R_sun**2 * sigma_sb))**0.25).to(u.K)

# Get out the temperature grid over each lat/lon from our exoplasim model
ts = earth.inspect("ts",tavg=True)
lon = earth.inspect("lon")
lat = earth.inspect("lat")
LON, LAT = np.meshgrid(lon, lat)

# Use 1 AU as present for Earth and convertt to meters for further calculations
a = 1 * u.AU
a_m = a.to(u.m)

fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

# Plotting conditions
plot_list = [("paper", 0.2), ("paper", 1.0), ("ideal", 0.2), ("ideal", 1.0)]

# Compute all photosynthesis rate maps to get global min and max
P_maps = []
for cond, f_a in plot_list:
    I = calculate_intensity_latlon(T_solar, R_solar, a_m, LAT, LON, f_a=f_a)
    P_map = calculate_P_curve(I, ts, conditions=cond)
    P_maps.append(P_map)

global_min = np.min([np.min(P) for P in P_maps])
global_max = np.max([np.max(P) for P in P_maps])

# Plot each photosynthesis rate map
pcm_list = []
for i, ax in enumerate(axes.flat):
    pcm = ax.pcolormesh(
        LON, LAT, P_maps[i], 
        cmap='RdBu_r', shading='gouraud',
        vmin=global_min, vmax=global_max
    )
    pcm_list.append(pcm)

    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_title(fr"Conditions: ${plot_list[i][0]}$, $f_a$: {plot_list[i][1]}")

# Shared colorbar on the right
cbar = fig.colorbar(pcm_list[0], ax=axes, 
                    label=r"Photosynthesis rate [$\mu$ mol photons $m^{-2}$ $s^{-1}$]",
                    location="right")

plt.suptitle("Planetary Photosynthesis Rate Map")
plt.show()

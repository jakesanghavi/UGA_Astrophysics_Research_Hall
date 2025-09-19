import exoplasim as exo
from astropy.constants import L_sun, R_sun, sigma_sb
from astropy import units as u
from utils import calculate_intensity_latlon, calculate_P_curve, get_planet_params, clean_name
import numpy as np
from matplotlib import pyplot as plt
import itertools

### SET THE PLANET NAME AND OTHER GLOBAL PARAMS HERE
planet_name = input("Input planet name: ")
N_YEARS = 2

# Parameter grid you are interested in seeing in the output plot
conditions_options = ["paper", "ideal"]
f_atm_options = [0.2, 1.0]

# Configure and run the exoplasim model for any given planet
# Must be supported in the utils file as the parameters are hard coded
# Currently supports only Earth and Trappist-1-e
planet_name_clean = clean_name(planet_name)
planet = exo.Model(workdir=f"planet_model_{planet_name_clean}",modelname=f"planet_model_{planet_name_clean}",resolution="T21",ncpus=4,layers=10,precision=8)

if planet_name_clean == "earth":  
    planet.configure()
else:
    parameters = get_planet_params(planet_name)
    planet.configure(**parameters)
    
planet.exportcfg()
planet.run(years=N_YEARS, clean=False)

# Configure parameters for the star of interest (the Sun here)
L_solar = L_sun.to(u.W)
R_solar = R_sun.to(u.m)
T_solar = ((L_sun / (4 * np.pi * R_sun**2 * sigma_sb))**0.25).to(u.K)

# Get out the temperature grid over each lat/lon from our exoplasim model
tavg = True if N_YEARS > 1 else False
ts = planet.inspect("ts",tavg=tavg)
lon = planet.inspect("lon")
lat = planet.inspect("lat")
LON, LAT = np.meshgrid(lon, lat)

# Use 1 AU as present for Earth and convertt to meters for further calculations
a = 1 * u.AU
a_m = a.to(u.m)

# Plotting conditions
plot_list = list(itertools.product(conditions_options, f_atm_options))

nrow = len({item[0] for item in plot_list})
ncol = len({item[1] for item in plot_list})

fig, axes = plt.subplots(nrow, ncol, figsize=(12, 10), constrained_layout=True)

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

plt.suptitle(f"Planetary Photosynthesis Rate Map - {planet_name}")

# Save the image
plt.savefig(f"photosynthesis_map_{planet_name_clean}_{N_YEARS}_years.png", dpi=300)
plt.show()

import exoplasim as exo
import numpy as np
from matplotlib import pyplot as plt

### PLANET CONFIGURATION ###
N_YEARS = 5
RESOLUTION = 'T21'
NCPUS = 4
NLAYERS = 10
PRECISION = 8
OUTPUT_TYPE = '.nc'
PLANET_NAME = 'EARTH'

# Vegetation settings
VEGETATION = 2
VEGACCEL = 1
INIT_GROWTH = 0.5
WET_SOIL = True

# Planet Comparison to Earth
PRESSURE_FRACTION = 1
MASS_RATIO = 1

def model_earthlike_stepwise(planet, year):
    planet.run(years=1, clean=False)
    veg = planet.inspect("vegplantc", tavg=True)

    lon = planet.inspect("lon")
    lat = planet.inspect("lat")

    # Have to open and close figure if doing it in a loop
    plt.figure(figsize=(8,4))
    plt.pcolormesh(lon, lat, veg, cmap='viridis', shading='gouraud')
    plt.colorbar(label="Vegetation rate")
    plt.xlabel("Longitude [deg]")
    plt.ylabel("Latitude [deg]")
    plt.title(f"Planetary Vegetation Rate Map - {PLANET_NAME} Scaled by {MASS_RATIO}")

    filename = f"vegetation_map_custom_earthlike_scaled_{MASS_RATIO}_year_{year}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    return planet


planet_params = {
        'vegetation': VEGETATION,
        'vegaccel': VEGACCEL,
        'initgrowth': INIT_GROWTH,
        'wetsoil': WET_SOIL,
        'pH2': 0.0,
        'pHe': 5.24e-6,
        'pN2': 0.78084,
        'pO2': 0.20946,
        'pCO2': 330.0e-6,
        'pAr': 9.34e-3,
        'pNe': 18.18e-6,
        'pKr': 1.14e-6,
        'pH2O': 0.01,
        'pCH4': 0.0
}

gas_params = ['pH2', 'pHe', 'pN2', 'pO2', 'pCO2', 'pAr', 'pNe', 'pKr', 'pH2O', 'pCH4']

for param in gas_params:
    planet_params[param] *= PRESSURE_FRACTION

g_new = 9.80665 * MASS_RATIO ** (1/3)
r_new = MASS_RATIO ** (1/3)

planet_params['gravity'] = g_new
planet_params['radius'] = r_new


# Create the model
planet = exo.Model(
    workdir="custom_earthlike_model",
    modelname="custom_earthlike_model",
    resolution=RESOLUTION,
    ncpus=NCPUS,
    layers=NLAYERS,
    precision=PRECISION,
    outputtype=OUTPUT_TYPE
)

# Configure and run
planet.configure(**planet_params)
planet.exportcfg()


for year in range(0, N_YEARS):
    planet = model_earthlike_stepwise(planet, year+1)

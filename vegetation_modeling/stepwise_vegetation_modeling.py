import exoplasim as exo
import numpy as np
from matplotlib import pyplot as plt
from veg_utils import calc_f_ret_big_rcb, calc_r_c
from constants import mearth

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
MASS_RATIO = 10

# Estimate radius of the planet based on its mass
# This is based on "The mass–radius relation of exoplanets revisited" by Müller et al. 2024
def piecewise_radius_estimate():
    # Small/rocky planets, like Earth
    if MASS_RATIO < 4.37:
        return 1.02 * (MASS_RATIO**0.27)
    # Intermediate-mass planets
    # H/He envelopes no longer neglible, so radius grows faster with mass than before
    if MASS_RATIO < 127:
        return 0.56 * (MASS_RATIO**0.67)
    # Massive planets, mass dominated by light gas.
    # Radius becomes almost constant and independent of mass
    # This gas is semi-degenerate, leading to the constant relation
    return 18.6 * (MASS_RATIO ** (-0.06))

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
    plt.title(
        f"Planetary Vegetation - Earthlike Planet Scaled by {MASS_RATIO}\n"
        f"Year: {year}"
    )

    filename = f"vegetation_map_custom_earthlike_scaled_{MASS_RATIO}_year_{year}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    return planet


planet_params = {
        'vegetation': VEGETATION,
        'vegaccel': VEGACCEL,
        'initgrowth': INIT_GROWTH,
        'wetsoil': WET_SOIL,
        'pHe': 5.24e-6,
}

gas_params = ['pH2', 'pHe', 'pN2', 'pO2', 'pCO2', 'pAr', 'pNe', 'pKr', 'pH2O', 'pCH4']

for param in gas_params:
    if param in planet_params:
        planet_params[param] *= PRESSURE_FRACTION

m_c = 0.325 * MASS_RATIO * mearth
r_c = calc_r_c(m_c)
r_rcb = 2 * r_c
t_eq = 255
planet_params['pHe'] = calc_f_ret_big_rcb(m_c, t_eq, r_c, r_rcb)

r_new = piecewise_radius_estimate()
g_new = 9.80665 * MASS_RATIO / (r_new ** 2)

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

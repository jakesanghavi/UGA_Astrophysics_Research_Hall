import exoplasim as exo
import numpy as np
from matplotlib import pyplot as plt
from veg_utils import calc_f_ret_big_rcb, calc_r_c, calc_r_B, calc_rho_rcb, calc_m_atm, calc_r_prime_b
from constants import mearth, Gsi, pi, rearth
import sys

### PLANET CONFIGURATION ###
N_YEARS = 1
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

# Gas settings
F_INIT = 0.15

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
    plt.pcolormesh(lon, lat, veg, cmap='viridis', shading='gouraud', vmin=0.0, vmax=0.03)
    plt.colorbar(label="Vegetation rate")
    plt.xlabel("Longitude [deg]")
    plt.ylabel("Latitude [deg]")
    plt.title(
        f"Planetary Vegetation - Earthlike Planet Scaled by {MASS_RATIO}\n"
        # f"Year: {year}"
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
    if param in planet_params:
        planet_params[param] *= PRESSURE_FRACTION

r_new = piecewise_radius_estimate()
g_new = 9.80665 * MASS_RATIO / (r_new ** 2)

planet_params['gravity'] = g_new
planet_params['radius'] = r_new
print(r_new)

# m_c = None
# m_atm = None
# if r_new <= 1.5 * rearth:
#     m_c = MASS_RATIO * mearth
#     m_atm = 8.6 * 10 **(-7) * MASS_RATIO * mearth
# else:
#     m_c = 0.95 * MASS_RATIO * mearth
#     m_atm = 0.03 * MASS_RATIO * mearth
m_c = MASS_RATIO * mearth
r_c = calc_r_c(m_c)
r_rcb = 2 * r_c
t_eq = 255
r_b = calc_r_B(m_c, t_eq)
rho_rcb = calc_rho_rcb(r_b, r_rcb)
r_prime_b = calc_r_prime_b(r_b)
# m_atm = calc_m_atm(rho_rcb, r_c, r_rcb, r_prime_b)
# F = m_atm/m_c
F_map = {(0.05, 1): 10 ** (-8), (0.05, 2): 0.001, (0.05, 5): 0.011, (0.05, 10): 0.014,
         (0.10, 1): 10 ** (-8), (0.10, 2): 0.005, (0.10, 5): 0.027, (0.10, 10): 0.038,
         (0.15, 1): 10 ** (-8), (0.15, 2): 0.011, (0.15, 5): 0.045, (0.15, 10): 0.064}
F = F_map[(F_INIT, MASS_RATIO)]
retained_frac = np.clip(calc_f_ret_big_rcb(m_c, t_eq, r_c, r_rcb), 0, 1)
planet_params['pHe'] = 0.25 * retained_frac * Gsi * F * (MASS_RATIO * mearth) ** 2 * 10 ** (-5)  / (4 * pi * (r_new * rearth) ** 4) * 10 **(-5)
planet_params['pH2'] = 0.75 * retained_frac * Gsi * F * (MASS_RATIO * mearth) ** 2 *  10 ** (-5) / (4 * pi * (r_new * rearth) ** 4) * 10 **(-5)

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

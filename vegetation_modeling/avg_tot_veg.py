import exoplasim as exo
import numpy as np
from matplotlib import pyplot as plt
from veg_utils import calc_f_ret_big_rcb, calc_r_c, calc_r_B, calc_rho_rcb, calc_m_atm, calc_r_prime_b
from constants import mearth, Gsi, pi, rearth
from atm_mass_frac import evolve_atmosphere
import sys
import json
import shutil
from wakepy import keep

### PLANET CONFIGURATION ###
N_YEARS = 1
RESOLUTION = 'T21'
NCPUS = 4
NLAYERS = 10
PRECISION = 8
OUTPUT_TYPE = '.nc'
PLANET_NAME = 'EARTH'
MIN_AU=0.75
MAX_AU=5
AU_STEP_SIZE=0.25

# Vegetation settings
VEGETATION = 2
VEGACCEL = 1
INIT_GROWTH = 0.5
WET_SOIL = True
BASE_FLUX = 1367

# Planet Comparison to Earth
PRESSURE_FRACTION = 1
MIN_MASS_RATIO = 0.5
MAX_MASS_RATIO = 8
MASS_STEP_SIZE = 0.25

# Gas settings
F_INIT = 0.15

# Estimate radius of the planet based on its mass
# This is based on "The mass–radius relation of exoplanets revisited" by Müller et al. 2024
def piecewise_radius_estimate(mass_ratio):
    # Small/rocky planets, like Earth
    if mass_ratio < 4.37:
        return 1.02 * (mass_ratio**0.27)
    # Intermediate-mass planets
    # H/He envelopes no longer neglible, so radius grows faster with mass than before
    if mass_ratio < 127:
        return 0.56 * (mass_ratio**0.67)
    # Massive planets, mass dominated by light gas.
    # Radius becomes almost constant and independent of mass
    # This gas is semi-degenerate, leading to the constant relation
    return 18.6 * (mass_ratio ** (-0.06))

def model_earthlike_stepwise(planet, year, mass_ratio):
    planet.run(years=1, clean=False)
    # veg = planet.inspect("vegplantc", tavg=True)
    

    # lon = planet.inspect("lon")
    # lat = planet.inspect("lat")

    # Have to open and close figure if doing it in a loop
    # plt.figure(figsize=(8,4))
    # plt.pcolormesh(lon, lat, veg, cmap='viridis', shading='gouraud', vmin=0.0, vmax=0.03)
    # plt.colorbar(label="Vegetation rate")
    # plt.xlabel("Longitude [deg]")
    # plt.ylabel("Latitude [deg]")
    # plt.title(
    #     f"Planetary Vegetation - Earthlike Planet Scaled by {mass_ratio}\n"
    #     # f"Year: {year}"
    # )

    # filename = f"vegetation_map_custom_earthlike_scaled_{mass_ratio}_year_{year}.png"
    # plt.savefig(filename, dpi=300, bbox_inches="tight")
    # plt.close()

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


def calculate_veg(mass_ratio, au):
    r_new = piecewise_radius_estimate(mass_ratio)
    g_new = 9.80665 * mass_ratio / (r_new ** 2)

    planet_params['gravity'] = g_new
    planet_params['radius'] = r_new
    planet_params['flux'] = BASE_FLUX / (au**2)
    planet_params['startemp'] = 5778

    m_c = mass_ratio * mearth
    r_c = calc_r_c(m_c)
    r_rcb = 2 * r_c
    t_eq = 255
    r_b = calc_r_B(m_c, t_eq)
    rho_rcb = calc_rho_rcb(r_b, r_rcb)
    r_prime_b = calc_r_prime_b(r_b)

    retained_frac = np.clip(calc_f_ret_big_rcb(m_c, t_eq, r_c, r_rcb), 0, 1)
    times, GCRs = evolve_atmosphere(
                Mc_me=mass_ratio,
                a_AU=au,
                t_disk_Myr=3.0,
                t_end_Gyr=5.0,
                init=F_INIT,
                dusty=True,
                eta=0.1,
                Lxuv0=1e22,
                t_sat_Myr=100,
                decay_index=1.1
            )

    # Targeting a time of 2 Gyr
    target_time = 2 * 10**9
    mask = times > target_time
    target_index = np.argmax(mask)
    F = GCRs[target_index]
    
    # If time is greater than max time, that means there has been no gas retained
    # as the simulation stops when M_atm == 0
    if target_index <= 0:
        F = 0
    planet_params['pHe'] = 0.25 * Gsi * F * (mass_ratio * mearth) ** 2 * 10 ** (-5)  / (4 * pi * (r_new * rearth) ** 4) * 10 **(-5)
    planet_params['pH2'] = 0.75 * Gsi * F * (mass_ratio * mearth) ** 2 *  10 ** (-5) / (4 * pi * (r_new * rearth) ** 4) * 10 **(-5)

    try:
        shutil.rmtree("custom_earthlike_model")
        shutil.rmtree("custom_earthlike_model_crashed")
    except:
        pass
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
        planet = model_earthlike_stepwise(planet, mass_ratio, year+1)
        
    veg = planet.inspect("vegplantc", tavg=True)
    land = planet.inspect('lsm')
    land = np.sum(land, axis=0)
    
    land_mask = land > 0
    masked_veg_values = veg[land_mask]
    average_veg = np.mean(masked_veg_values)
    tot_veg = np.sum(masked_veg_values)
        
    return [average_veg, tot_veg]
        
output_dict = {}
with keep.presenting():
    masses = np.arange(MIN_MASS_RATIO, MAX_MASS_RATIO, MASS_STEP_SIZE)
    aus = np.arange(MIN_AU, MAX_AU, AU_STEP_SIZE)
    for mr in masses:
        for au in aus:
            try:
                veg_amt = calculate_veg(mr, au)
                output_dict[str(mr)] = [float(v) for v in veg_amt]
                with open(f"veg_json_FI_{F_INIT}_MS_{MASS_STEP_SIZE}_AUS_{AU_STEP_SIZE}.json", "w") as f:
                    json.dump(output_dict, f, indent=4)
            except Exception as e:
                print(f"Error!: {e}")
                print(mr)
                sys.exit()
    
print(output_dict)

import exoplasim as exo
from astropy.constants import L_sun, R_sun, sigma_sb
from astropy import units as u
import numpy as np
from matplotlib import pyplot as plt

### PLANET CONFIGURATION ###

N_YEARS = 3
RESOLUTION = 'T21'
NCPUS = 4
NLAYERS = 10
PRECISION = 8
OUTPUT_TYPE = '.nc'

# Vegetation settings
VEGETATION = 2
VEGACCEL = 1
INIT_GROWTH = 0.5
WET_SOIL = True

# Mirror Earth but with lower atmospheric pressure
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

# Create the model
planet = exo.Model(
    workdir="model_earth_manual",
    modelname="model_earth_manual",
    resolution=RESOLUTION,
    ncpus=NCPUS,
    layers=NLAYERS,
    precision=PRECISION,
    outputtype=OUTPUT_TYPE
)

# Configure and run
planet.configure(**planet_params)
planet.exportcfg()
planet.run(years=N_YEARS, clean=False)

# Analyze vegetation
veg = planet.inspect("veggpp", tavg=True)
land = planet.inspect("lsm")

land = np.sum(land, axis=0)
land_mask = land > 0

masked_veg_values = veg[land_mask]
average_veg = np.mean(masked_veg_values)
tot_veg = np.sum(masked_veg_values)

# Save results
with open(f"vegetation_results_{N_YEARS}.txt", "w") as f:
    f.write(f"Average Vegetation: {average_veg}\n")
    f.write(f"Total Vegetation: {tot_veg}\n")
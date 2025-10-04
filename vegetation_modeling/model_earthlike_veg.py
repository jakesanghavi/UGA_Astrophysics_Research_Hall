import exoplasim as exo
import numpy as np
from matplotlib import pyplot as plt

### PLANET CONFIGURATION ###
N_YEARS = 2
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

# Planet Comparison to Earth
PRESSURE_FRACTION = 1
MASS_RATIO = 1

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

def model_earthlike(pressure_fraction=1, mass_ratio=1):
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
        planet_params[param] *= pressure_fraction
    
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
    planet.run(years=N_YEARS, clean=False)
    
    return planet

planet = model_earthlike(PRESSURE_FRACTION, MASS_RATIO)

# Get out the temperature grid over each lat/lon from our exoplasim model
veg = planet.inspect("vegplantc",tavg=True)
lon = planet.inspect("lon")
lat = planet.inspect("lat")
LON, LAT = np.meshgrid(lon, lat)

plt.pcolormesh(lon,lat,veg, cmap='viridis',shading='Gouraud')
plt.colorbar()
plt.xlabel("Longitude [deg]")
plt.ylabel("Latitude [deg]")
plt.title(
    f'Planetary Vegetation Rate Map - Low Pressure "Earth"\n'
    f"Year: {N_YEARS}"
)

# Save the image, if desired
# plt.savefig(f"vegetation_map_custom_earthlike_{VEGETATION}_vegetation_{VEGACCEL}_vegaccel_\
#     {N_YEARS}_years.png", dpi=300)
plt.show()

import exoplasim as exo
from astropy.constants import L_sun, R_sun, sigma_sb
from astropy import units as u
from utils import calculate_intensity_latlon, calculate_P_curve, get_planet_params, clean_name
import numpy as np
from matplotlib import pyplot as plt
import itertools

### SET THE PLANET NAME AND OTHER GLOBAL PARAMS HERE
planet_name = input("Input planet name: ")
N_YEARS = 1
RESOLUTION = 'T21'
NCPUS = 4
NLAYERS = 10
PRECISION = 8
OUTPUT_TYPE = '.nc'

# 0 is none, 1 is static, 2 is dynamic
VEGETATION = 2
VEGACCEL = 1
INIT_GROWTH = 0.5
WET_SOIL = True

# Configure and run the exoplasim model for any given planet
# Must be supported in the utils file as the parameters are hard coded
# Currently supports only Earth and Trappist-1-e
planet_name_clean = clean_name(planet_name)
planet = exo.Model(workdir=f"planet_model_{planet_name_clean}",modelname=f"planet_model_{planet_name_clean}", resolution=RESOLUTION, ncpus=NCPUS, layers=NLAYERS, precision=PRECISION, outputtype=OUTPUT_TYPE)

if planet_name_clean == "earth":  
    planet.configure(**{"vegetation": VEGETATION, "vegaccel": VEGACCEL, 
                        "initgrowth": INIT_GROWTH, "wetsoil": WET_SOIL})
else:
    parameters = get_planet_params(planet_name)
    parameters['vegetation'] = VEGETATION
    parameters['vegaccel'] = VEGACCEL
    parameters['initgrowth'] = INIT_GROWTH
    parameters['wetsoil'] = WET_SOIL
    planet.configure(**parameters)
    
planet.exportcfg()
planet.run(years=N_YEARS, clean=False)

# Get out the temperature grid over each lat/lon from our exoplasim model
veg = planet.inspect("vegplantc",tavg=True)
lon = planet.inspect("lon")
lat = planet.inspect("lat")
LON, LAT = np.meshgrid(lon, lat)

plt.pcolormesh(lon,lat,veg,cmap='Greens',shading='Gouraud')
plt.colorbar()
plt.xlabel("Longitude [deg]")
plt.ylabel("Latitude [deg]")
plt.title(
    f"Planetary Vegetation Rate Map - {planet_name}\n"
    f"Vegetation Acceleration Factor: {VEGACCEL}"
)

# Save the image
plt.savefig(f"vegetation_map_{planet_name_clean}_{VEGETATION}_vegetation_{VEGACCEL}_vegaccel_\
    {N_YEARS}_years.png", dpi=300)
plt.show()

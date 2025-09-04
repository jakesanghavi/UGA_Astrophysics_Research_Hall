import numpy as np
from astropy.constants import GM_sun, GM_earth, GM_jup, R_sun, R_earth, R_jup, L_sun, sigma_sb, k_B as akb, c as ac, h, G
import astropy.units as u

pi = np.pi
twopi = 2.0 * pi
piby2 = 0.5 * pi

# Time units
# Inexact, can be changed later
year = 3.1556926e7
yearInDays = 365.24
degToRad = pi / 180.0
radToDeg = 1.0 / degToRad

# Distance units
AU = (1 * u.au).value

# Mass units
msol = GM_sun.value / G.value
mearth = GM_earth.value / G.value
mjup = GM_jup.value / G.value

# Radius units
rsol = R_sun.value
rearth = R_earth.value
rjup = R_jup.value

# Luminosity
lsol = L_sun.value

# Conversion factors
msolToMEarth = msol / mearth
solradToAU = rsol / AU
solradToREarth = rsol / rearth

# Gravitational constants
# SI
Gsi = G.value
# Solar mass-AU units (time units: 2*pi units = 1 year)
Gmau = 1.0             
# Solar mass-AU-day units       
Gmau_day = 2.959e-4           

# Solar flux
# W/m^2
fluxsol = lsol / (4.0 * pi * AU * AU)
# erg/s/cm^2
fluxsolcgs = fluxsol * 1000              

# Physical constants
hplanck = h.value       
stefan = 5.67e-8 
c = ac.value   
k_B = akb.value
sigma_SB = sigma_sb.value
# Speed of light in cm/s
c_cgs = c*100                
c_mau = c_cgs * year / (twopi * AU)

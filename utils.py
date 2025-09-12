import numpy as np
from astropy.constants import sigma_sb, h, c, k_B, N_A
from astropy import units as u
from scipy.integrate import quad
import re

def calculate_max_photosynthesis_rate_no_resp():
    """
    Calculate the maximum photosynthesis rate in units
    mu mol photons m^(-2) s^(-1) based on parameter
    estimates found in the paper.
    Parameters
    ----------
    None
    
    Returns
    float : the max. rate
    """
    alpha = 1 * 10 ** (-5)
    beta = 1 * 10 ** (-3)
    gamma = 2.0
    
    return 1/(beta+2*(alpha*gamma)**(1/2))

def calculate_resp(conditions="ideal"):
    """
    Calculate the dark respiration rate in units
    mu mol photons m^(-2) s^(-1) as a fraction of
    the max. photosynthesis rate.
    Parameters
    ----------
    conditions: String
        One of ['ideal', 'optimistic', 'pessimistic']
    
    Returns
    float : the dark respiration rate
    """
    
    if conditions == 'paper':
        return 20.0
    
    max_no_resp = calculate_max_photosynthesis_rate_no_resp()
    
    rate_fractions = {
        "ideal": 0.3,
        "optimistic": 0.6,
        "pessimistic": 0.8
    }
    
    return max_no_resp * rate_fractions[conditions]

def calculate_photosynthesis_rate(I, conditions="ideal"):
    """
    Calculate the photosynthesis rate in units
    mu mol photons m^(-2) s^(-1).
    Parameters
    ----------
    I : float
        The Irradiance intensity in units W m^(-2)
    conditions : String
        One of ['ideal', 'optimistic', 'pessimistic']
    
    Returns
    float : the photosynthesis rate
    """
    # I_val = I.to_value(u.W / u.m**2)
    I_val = I
    alpha = 1 * 10 ** (-5)
    beta = 1 * 10 ** (-3)
    gamma = 2.0
    
    resp = calculate_resp(conditions)
    
    return I_val / (alpha * I_val ** 2 + beta * I_val + gamma) - resp

# NOT CURRENTLY BEING USED
# REPLACED BY EXOPLASIM
def calculate_T_equilibrium(L, a):
    """
    Calculate the equilibrium temperature of a planet based on 
    equation 9 in the paper.
    Parameters
    ----------
    L : float
        The luminosity of the parent star in units W
    a: float
        The semi-major axis of the orbit around the parent star
        in units m
    
    Returns
    float : the equilibrium temperature of the planet in units K
    """
    A_bond = 0.306
    numerator = L * (1-A_bond)
    denominator = 16 * sigma_sb * np.pi * a ** 2
    return (numerator/denominator) ** (1/4)

# NOT CURRENTLY IMPLEMENTED
# ALWAYS RETURNS 0
def calculate_delta_T_greenhouse():
    """
    Calculate the impact of greenhouse effects on 
    the temperature of the planet of interest.
    Parameters
    ----------
    None
    
    Returns
    float : the temperature impact
    """
    return 0 * u.K

# NOT CURRENTLY BEING USED
def calculate_f_temp(L, a):
    """
    Calculate the f_temp.
    Parameters
    ----------
    L : float
        The luminosity of the parent star in units W
    a: float
        The semi-major axis of the orbit around the parent star
        in units m
    
    Returns
    float : the f_temp in units K
    """
    T_eq = calculate_T_equilibrium(L, a)
    delta_T_greenhouse = calculate_delta_T_greenhouse()
    T = T_eq + delta_T_greenhouse
        
    T_opt = (35 * u.deg_C).to(u.K, equivalencies=u.temperature())
    T_max = (73 * u.deg_C).to(u.K, equivalencies=u.temperature())
    
    term1 = ((T_max - T)/(T_max - T_opt))
    term2 = (T/T_opt)
        
    return (term1 * term2) ** (T_opt/(T_max-T_opt))

def calculate_f_temp_given_T(T):
    """
    Calculate the f_temp given a surface temperature.
    These surface temperatures will be provided by exoplasim
    Parameters
    ----------
    T: float
        the surface temperature in units K
    
    Returns
    float : the f_temp in units K
    """
    T = T * u.K
    T = T + calculate_delta_T_greenhouse()

    T = T.to(u.deg_C, equivalencies=u.temperature())
    T_opt = 35 * u.deg_C
    T_max = 73 * u.deg_C
    
    term1 = ((T_max - T)/(T_max - T_opt))
    term2 = (T/T_opt)
    base = term1 * term2
    
    # Avoid NaN problems
    base = np.clip(base, 0, None)
    
    output = base ** (T_opt/(T_max-T_opt))
    return output

def calculate_P_curve(I, T, conditions="ideal"):
    """
    Calculate the photosynthesis rate of a planet
    given both irradiance intensity data and 
    temperature data.
    Parameters
    ----------
    I : float
        The Irradiance intensity in units W m^(-2)
    T: float
        The surface temperature of the planet in units K
    conditions: String
        One of ['ideal', 'optimistic', 'pessimistic'] 
    
    Returns
    float : the photosynthesis rate
    """
    f_temp = calculate_f_temp_given_T(T)
    return f_temp * calculate_photosynthesis_rate(I, conditions=conditions)

def par_integral(lam, T_star):
    """
    Calculate the photon radiance of a star at a given wavelength.
    This will be used to integrate over a range of wavelengths
    Parameters
    ----------
    lam : float
        Wavelength in meters
    T_star : float
        Temperature of the star in Kelvin
    
    Returns
    -------
    float : Photon flux density at the given wavelength
    """
    numerator = (2 * c.value / lam**4)
    denominator = (np.exp(h.value * c.value / (lam * k_B.value * T_star.value)) - 1)
    return numerator/denominator

def photon_flux_star(T_star, R_star):
    """
    Integrate the photon radiance of a star
    across the photosynthetically active radiation (PAR)
    wavelength range to calculate the total photon flux
    emitted by the star.

    Parameters
    ----------
    T_star : float
        Temperature of the star in Kelvin
    R_star : float
        Radius of the star in meters

    Returns
    -------
    float : Total photon flux emitted by the star in the PAR range
    """
    lambda_min = 400 * 10 ** (-9)
    lambda_max = 700 * 10 ** (-9)
    integral, _ = quad(par_integral, lambda_min, lambda_max, args=(T_star,), limit=200)
    return 4 * np.pi * R_star**2 * integral

def photon_flux_at_planet(T_star, R_star, a):
    """
    Calculate the photon flux received by a planet
    from its parent star at orbital distance a.
    (Only over the PAR wavelengths)

    Parameters
    ----------
    T_star : float
        Temperature of the star in Kelvin
    R_star : float
        Radius of the star in meters
    a : float
        Orbital semi-major axis (distance from the star) in meters

    Returns
    -------
    float : Photon flux at the planet’s orbit in the PAR range
    """
    n_dot = photon_flux_star(T_star, R_star)
    return n_dot / (4 * np.pi * a**2)

def calculate_intensity(T_star, R_star, a, f_a=1.0):
    """
    Calculate the photosynthesically active radiation (PAR)
    for a planet, given the parameters of itself and 
    its parent star.
    Parameters
    ----------
    T_star : float
        The temperature of the parent star in units K
    R_star: float
        The radius of the parent star in units m
    a: float
        The semi-major axis of orbit in units m
    f_atm: float
        The atmospheric attenuation on the planet
    
    Returns
    float : the intensity, corrected by atmospheric
            attenuation
    """
    conversion_factor = 1e6 / N_A.value
    flux = photon_flux_at_planet(T_star, R_star, a)
    return f_a * flux * conversion_factor

# Solve Kepler's Equation: M = E - e*sin(E) for E
def kepler_equation(E, M, e):
    return E - e * np.sin(E) - M

def kepler_equation_prime(E, e):
    return 1 - e * np.cos(E)

def orbital_elements_to_state_vectors(semimaj, ecc, inc, longascend, argper, meananom, G, total_mass):
    """Convert orbital elements into Cartesian position and velocity vectors (in AU, AU/day or chosen units)."""

    mu = G * total_mass  # gravitational parameter

    # Newton-Raphson iteration
    E = meananom
    for _ in range(100):
        dE = -kepler_equation(E, meananom, ecc) / kepler_equation_prime(E, ecc)
        E += dE
        if abs(dE) < 1e-10:
            break

    # True anomaly
    nu = 2 * np.arctan2(
        np.sqrt(1 + ecc) * np.sin(E / 2),
        np.sqrt(1 - ecc) * np.cos(E / 2),
    )

    # Distance
    r = semimaj * (1 - ecc * np.cos(E))

    # Position in orbital plane
    x_orb = r * np.cos(nu)
    y_orb = r * np.sin(nu)
    z_orb = 0.0

    # Velocity in orbital plane
    # Safe divide by r
    if r != 0.0:
        rdot = (np.sqrt(mu * semimaj) / r) * (-np.sin(E))
        rfdot = (np.sqrt(mu * semimaj) / r) * (np.sqrt(1 - ecc**2) * np.cos(E))
    else:
        rdot, rfdot = 0.0, 0.0

    vx_orb = rdot * np.cos(nu) - rfdot * np.sin(nu)
    vy_orb = rdot * np.sin(nu) + rfdot * np.cos(nu)
    vz_orb = 0.0

    # Rotation matrices for inclination, longitude of ascending node, argument of periapsis
    cosO, sinO = np.cos(longascend), np.sin(longascend)
    cosi, sini = np.cos(inc), np.sin(inc)
    cosw, sinw = np.cos(argper), np.sin(argper)

    R = np.array([
        [cosO*cosw - sinO*sinw*cosi, -cosO*sinw - sinO*cosw*cosi, sinO*sini],
        [sinO*cosw + cosO*sinw*cosi, -sinO*sinw + cosO*cosw*cosi, -cosO*sini],
        [sinw*sini, cosw*sini, cosi]
    ])

    r_vec = R @ np.array([x_orb, y_orb, z_orb])
    v_vec = R @ np.array([vx_orb, vy_orb, vz_orb])

    return r_vec, v_vec

def safe_get(getter, key, index, default=None):
    try:
        return getter(key, index)
    except KeyError:
        return default
    
def calculate_intensity_latlon(T_star, R_star, a, lat, lon, f_a=1.0, declination=0.0):
    """
    Calculate PAR intensity using star parameters, orbit parameters, and lat/long.
    Parameters
    ----------
    T_star : float
        Star temperature in K
    R_star : float
        Star radius in m
    a : float
        Orbital semi-major axis in m
    lat : array
        Latitude grid in degrees
    lon : array
        Longitude grid degrees
    f_a : float
        Atmospheric attenuation factor
    declination : float
        Setting to 0
    Returns
    -------
    I : array
        PAR intensity grid
    """
    # Base flux at top of atmosphere (old calculation)
    base_flux = photon_flux_at_planet(T_star, R_star, a)
    conversion_factor = 1e6 / N_A.value
    base_intensity = base_flux * conversion_factor * f_a
    
    # Convert to radians
    # Longitude is not being used right now - need advice on how to implement
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    dec_rad = np.radians(declination)
    
    # Local zenith angle factor
    cos_theta = np.cos(lat_rad) * np.cos(dec_rad)
    cos_theta = np.clip(cos_theta, 0, None)
    
    ### SEASONAL FACTOR NOT IMPLEMENTED
    
    return base_intensity * cos_theta

def get_planet_params(planet_name):
    """
    Calculate PAR intensity using star parameters, orbit parameters, and lat/long.
    Parameters
    ----------
    planet_name : string
        Name of the planet
    Returns
    -------
    params : dict
        Parameter grid for the given planet
    """
    clean_body_name = clean_name(planet_name)
    if clean_body_name == "trappist1e":
        params = {
                "startemp": 2566.0,
                "flux": 900.0,
                "eccentricity": 0.005,
                "obliquity": 0.0,
                "fixedorbit": True,
                "synchronous": True,
                "rotationperiod": 6.101,
                "radius": 0.92,
                "gravity": 9.11,
                "aquaplanet": False,
                "timestep": 30.0,
                "snapshots": 720,
                "physicsfilter": "gp|exp|sp"
                }
        
        return params
    else:
        raise ValueError("Planet not yet supported!")
    
def clean_name(body_name):
    """
    Normalizes the name of a body
    Parameters
    ----------
    body_name : string
        The name of the body
    Returns
    -------
    cleaned_string : string
        Cleaned name of the body
    """
    # Use re.sub() to remove non-alphanumeric characters, then convert to lowercase
    cleaned_string = re.sub(r'[^a-zA-Z0-9]', '', body_name).lower()
    return cleaned_string
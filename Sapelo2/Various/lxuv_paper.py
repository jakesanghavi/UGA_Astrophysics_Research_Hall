"""
============================================================
Atmosphere Evolution Model (Bondi-limited core-powered loss)
With Localized Paper-Derived Wind Profile (Picogna et al. 2019)
------------------------------------------------------------
- Core radius (Zeng scaling)
- Nebular accretion (Lee & Chiang-like)
- Boil-off (Owen & Wu heuristic)
- Core-powered mass loss (Ginzburg et al.)
- XUV photoevaporation (Paper-based Local Surface Density Profile)
- Bondi-limited mass loss: spontaneous loss stops when t_cool > t_Bondi
============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import G, k, m_p, sigma
from math import pi

# ============================================================
# Constants
# ============================================================
M_earth = 5.9722e24
R_earth = 6.371e6
year = 365.25 * 24 * 3600
AU = 1.496e11
L_sun = 3.828e26
M_sun = 1.98847e30

def get_lxuv0_from_bolometric(L_bol_present):
    """
    Calculates the initial saturated XUV luminosity (Lxuv0) directly from 
    the star's present-day overall bolometric luminosity.
    
    Parameters:
    -----------
    L_bol_present : float
        The overall total luminosity of the star (modern day).
    unit : str
        The unit of L_bol_present. Options: 'watts' (SI) or 'solar' (L_sun).
        
    Returns:
    --------
    Lxuv0_watts : float
        The initial saturated high-energy baseline in Watts.
    """
    # If passed in solar units, convert to Watts (SI)
    L_bol_watts = L_bol_present

    # Young stars saturate their corona at ~0.1% of their total bolometric budget
    f_sat = 10**(-2.7)
    Lxuv0_watts = L_bol_watts * f_sat
    
    return Lxuv0_watts

# ============================================================
# Core radius
# ============================================================
def core_radius_zeng(Mc_mearth, CMF=0.26):
    return R_earth * (1.07 - 0.21 * CMF) * (Mc_mearth ** (1/3.7))

# ============================================================
# Initial envelope (Lee & Chiang)
# ============================================================
def initial_GCR(init=0.05):
    return init

# ============================================================
# Boil-off (Owen & Wu heuristic)
# ============================================================
def apply_boiloff(Matm, Mc_kg, Rc, T_eq, mu=2.3):
    g = G * Mc_kg / Rc**2
    H = k * T_eq / (mu * m_p * g)
    inflation = H / Rc
    if inflation < 1e-3:
        return Matm
    if inflation > 0.05:
        retain = 0.1
    elif inflation > 0.02:
        retain = 0.3
    else:
        retain = 0.6
    return Matm * retain

# ============================================================
# Cooling luminosity
# ============================================================
def cooling_luminosity_PY(Matm, Mc_kg, Rc, mu=29, kappa=0.1, gamma=7/5):
    # Gamma 5/3 is monatomic (ex. helium), Gamma = 7/5 for diatomic (ex. hydrogen)
    
    nabla_ad = (gamma - 1) / gamma
    E_env = G * Mc_kg * Matm / Rc

    # RCB temperature from ideal-gas + hydrostatic balance
    T_rcb = (G * Mc_kg * mu * m_p) / (k * Rc)

    # --- 3. RCB pressure from optical depth ~ 1 condition
    # τ ~ κ P / g ~ 1 → P ~ g / κ
    g = G * Mc_kg / Rc**2
    P_rcb = g / kappa

    # Radiative diffusion luminosity at RCB
    L = (64 * 3.1416 * G * Mc_kg * sigma * T_rcb**4 /
         (3 * kappa * P_rcb)) * nabla_ad

    # Cooling time
    t_cool = E_env / L

    return L, t_cool

# ============================================================
# Core-powered mass loss
# ============================================================
def core_powered_loss(L, g, Rc):
    return L / (g * Rc)

# ============================================================
# Bondi-limited timescale
# ============================================================
def bondi_timescale(Matm, Mc_kg, Rp, T, mu=29):
    cs = np.sqrt(k * T / (mu * m_p))
    R_B = G * Mc_kg / cs**2
    rho = Matm / (4/3 * pi * Rp**3)
    mdot_B = 4 * pi * R_B**2 * rho * cs
    return Matm / max(mdot_B, 1e-20)

# ============================================================
# Stellar XUV track
# ============================================================
def Lxuv_track(t, Lxuv0=1e22, t_sat=100e6*year, decay_index=1.1):
    return Lxuv0 if t <= t_sat else Lxuv0 * (t / t_sat) ** (-decay_index)

# ============================================================
# Paper-Based Photoevaporation Profile (Equation 2 Local Flux)
# ============================================================
def photoevaporation_rate_paper(a_AU, L_X_watts, Rp_meters):
    """
    Calculates the local atmospheric mass loss by finding the wind surface 
    density Sigma_w(R) at the planet's exact radius using Equation 2, 
    then multiplying by the planet's geometric capture area.
    """
    # 1. Base Normalization from Equation 5 (M_sun/yr)
    L_X_ergs = L_X_watts * 1e7
    A_L, B_L, C_L, D_L = -2.7326, 3.3307, -2.9868e-3, -7.2580
    log10_LX = max(np.log10(L_X_ergs), 20.0)
    log10_M_dot_Lx = A_L * np.exp(((np.log(log10_LX) - B_L) ** 2) / C_L) + D_L
    M_dot_Lx = 10 ** log10_M_dot_Lx

    # 2. Polynomial Profile Coefficients (Eq 3 parameters)
    a, b, c, d, e, f, g = -0.5885, 4.3130, -12.1214, 16.3587, -11.4721, 5.7248, -2.8562
    logR = np.log10(a_AU)
    
    # Value of the base polynomial fit
    poly_val = (a*logR**6 + b*logR**5 + c*logR**4 + d*logR**3 + e*logR**2 + f*logR + g)
    
    # Derivative component of the polynomial: d(poly)/d(logR)
    d_poly = (6*a*logR**5 + 5*b*logR**4 + 4*c*logR**3 + 3*d*logR**2 + 2*e*logR + f)

    # Calculate local wind surface density Sigma_w(R) [M_sun / AU^2 / yr] using Equation 2
    # Guard against zero or negative slope regions away from the core profile peak
    sigma_w = (M_dot_Lx * (10**poly_val) / (2 * pi * a_AU**2)) * (d_poly / a_AU)
    sigma_w = max(sigma_w, 1e-25)

    # 3. Convert Surface Density to SI units [kg / m^2 / s]
    sigma_w_si = (sigma_w * M_sun) / (year * AU**2)

    # 4. Calculate local capture rate across the planet's physical cross-section (pi * Rp^2)
    mdot_local_kg_s = sigma_w_si * (pi * Rp_meters**2)
    
    return mdot_local_kg_s

# ============================================================
# Planet radius
# ============================================================
def planet_radius(Rc, Mc_kg, Matm):
    f = Matm / Mc_kg
    return Rc * (1 + 30 * f**0.25)

# ============================================================
# Atmosphere evolution
# ============================================================
def evolve_atmosphere(Mc_me, a_AU=0.1, t_disk_Myr=3.0, t_end_Gyr=5.0,
                      Z=0.02, dusty=True, init=0.05, mu=2.2, eta=0.1,
                      Lxuv0=3e22, t_sat_Myr=100, decay_index=1.1,
                      show_progress=False):

    Mc_kg = Mc_me * M_earth
    Rc = core_radius_zeng(Mc_me)
    T_eq = (L_sun / (16 * pi * (a_AU*AU)**2 * sigma))**0.25

    # Initial envelope
    GCR0 = initial_GCR(init)
    Matm = apply_boiloff(GCR0*Mc_kg, Mc_kg, Rc, T_eq, mu)

    t = t_disk_Myr * 1e6 * year
    t_end = t_end_Gyr * 1e9 * year
    times, GCRs = [t/year], [Matm/Mc_kg]
    xuvs = [Lxuv0]

    while t < t_end and Matm > 0:
        M_p = Mc_kg + Matm
        Rp = planet_radius(Rc, Mc_kg, Matm)
        g_acc = G * Mc_kg / Rc**2

        # Cooling and Bondi times
        L, t_cool = cooling_luminosity(Matm, Mc_kg, Rc)
        t_B = bondi_timescale(Matm, Mc_kg, Rp, T_eq, mu)

        # Core-powered loss
        if t_cool < t_B:
            mdot_core = core_powered_loss(L, g_acc, Rc)
        else:
            mdot_core = 0.0 

        # Local XUV photoevaporative wind interaction
        L_XUV = Lxuv_track(t, Lxuv0, t_sat_Myr * 1e6 * year, decay_index)
        mdot_xuv = photoevaporation_rate_paper(a_AU, L_XUV, Rp)

        # Total mass loss
        mdot = mdot_core + mdot_xuv
        if mdot <= 0: 
            break

        dt = min(1e7*year, max(1e2*year, 0.01*Matm/mdot))
        Matm = max(Matm - mdot*dt, 0)
        t += dt

        if show_progress and len(times) % 200 == 0:
            print(f"t={t/year:.2e} yr, GCR={Matm/Mc_kg:.3e}")

        times.append(t/year)
        GCRs.append(Matm/Mc_kg)
        xuvs.append(L_XUV)

    return np.array(times), np.array(GCRs), np.array(xuvs)

# ============================================================
# Run and plot
# ============================================================
def run_planet_model(Mc_me=1.0, a_AU=2.0, t_disk_Myr=3.0, t_end_Gyr=5.0,
                     Z=0.02, dusty=True, init=0.05, mu=2.2, eta=0.1, Lnow=3e22,
                     t_sat_Myr=100, decay_index=1.1):

    Lxuv0 = get_lxuv0_from_bolometric(Lnow)
    times, GCRs, xuvs = evolve_atmosphere(Mc_me, a_AU, t_disk_Myr, t_end_Gyr,
                                    Z, dusty, init, mu, eta,
                                    Lxuv0, t_sat_Myr, decay_index)

    plt.figure(figsize=(7, 4))
    plt.loglog(times/1e6, GCRs, color='darkcyan', label='Equation 2 Local Wind Logic')
    plt.xlabel("Time (Myr)")
    plt.ylabel("Atmosphere mass fraction $M_{atm}/M_c$")
    plt.title(f"{Mc_me} M⊕ planet at {a_AU} AU")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"Final GCR at {t_end_Gyr} Gyr: {GCRs[-1]:.3e}")
    return times, GCRs, xuvs

# ============================================================
# Call to run
# ============================================================
if __name__ == "__main__":
    run_planet_model(
        Mc_me=1.0,   # 1 Earth Mass
        a_AU=1.0,    # 2 AU
        t_disk_Myr=3.0,
        t_end_Gyr=5.0,
        init=0.01,
        dusty=True,
        eta=0.1,
        Lnow=3.828e26, 
        t_sat_Myr=100,
        decay_index=1.1
    )
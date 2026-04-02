"""
============================================================
Atmosphere Evolution Model (Bondi-limited core-powered loss)
------------------------------------------------------------
- Core radius (Zeng scaling)
- Nebular accretion (Lee & Chiang-like)
- Boil-off (Owen & Wu heuristic)
- Core-powered mass loss (Ginzburg et al.)
- XUV photoevaporation (Erkaev energy-limited)
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

# ============================================================
# Core radius
# ============================================================
def core_radius_zeng(Mc_mearth, CMF=0.26):
    return R_earth * (1.07 - 0.21 * CMF) * (Mc_mearth ** (1/3.7))

# ============================================================
# Roche-lobe factor
# ============================================================
def roche_factor(xi):
    return 1.0 if xi <= 1 else 1 - 1.5/xi + 0.5/xi**3

# ============================================================
# Stellar XUV track
# ============================================================
def Lxuv_track(t, Lxuv0=1e22, t_sat=100e6*year, decay_index=1.1):
    return Lxuv0 if t <= t_sat else Lxuv0 * (t / t_sat) ** (-decay_index)

# ============================================================
# Initial envelope (Lee & Chiang)
# ============================================================
# Mc_mearth, t_disk_Myr=3.0, Z=0.02, dusty=True
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
def cooling_luminosity(Matm, Mc_kg, Rc):
    E_bind = G * Mc_kg * Matm / Rc
    GCR = Matm / Mc_kg
    t_KH = 1e7 * year * (max(GCR,1e-4)/0.01)**(-3)
    return E_bind / t_KH, t_KH

# ============================================================
# Core-powered mass loss
# ============================================================
def core_powered_loss(L, g, Rc):
    return L / (g * Rc)

# ============================================================
# Bondi-limited timescale
# ============================================================
def bondi_timescale(Matm, Mc_kg, Rp, T, mu=2.2):
    cs = np.sqrt(k * T / (mu * m_p))
    R_B = G * Mc_kg / cs**2
    rho = Matm / (4/3 * pi * Rp**3)
    mdot_B = 4 * pi * R_B**2 * rho * cs
    return Matm / max(mdot_B,1e-20)

# ============================================================
# XUV mass loss
# ============================================================
def photoevaporation_rate(Rp, M_p, a_AU, L_XUV, eta=0.1):
    d = a_AU * AU
    F_xuv = L_XUV / (4 * pi * d**2)
    q = M_p / M_sun
    R_L = d * 0.49 * q**(1/3)
    xi = max(R_L / Rp,1)
    K = roche_factor(xi)
    return eta * pi * Rp**2 * F_xuv / (G * M_p / Rp) / K

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
    # GCR0 = initial_GCR(Mc_me, t_disk_Myr, Z, dusty)
    GCR0=initial_GCR(init)
    Matm = apply_boiloff(GCR0*Mc_kg, Mc_kg, Rc, T_eq, mu)

    t = t_disk_Myr*1e6*year
    t_end = t_end_Gyr*1e9*year
    times, GCRs = [t/year], [Matm/Mc_kg]

    while t < t_end and Matm > 0:
        M_p = Mc_kg + Matm
        Rp = planet_radius(Rc, Mc_kg, Matm)
        g = G * Mc_kg / Rc**2

        # Cooling and Bondi times
        L, t_cool = cooling_luminosity(Matm, Mc_kg, Rc)
        t_B = bondi_timescale(Matm, Mc_kg, Rp, T_eq, mu)

        # Core-powered loss only if faster than Bondi limit
        if t_cool < t_B:
            mdot_core = core_powered_loss(L, g, Rc)
        # Bondi-limited regime - should stop spontaneous loss
        # Doesn't seem to work though
        else:
            mdot_core = 0.0 

        # XUV-driven loss continues
        L_XUV = Lxuv_track(t, Lxuv0, t_sat_Myr*1e6*year, decay_index)
        mdot_xuv = photoevaporation_rate(Rp, M_p, a_AU, L_XUV, eta)

        # Total mass loss
        mdot = mdot_core + mdot_xuv
        if mdot <= 0: break

        dt = min(1e7*year, max(1e2*year, 0.01*Matm/mdot))
        Matm = max(Matm - mdot*dt, 0)
        t += dt

        if show_progress and len(times) % 200 == 0:
            print(f"t={t/year:.2e} yr, GCR={Matm/Mc_kg:.3e}")

        times.append(t/year)
        GCRs.append(Matm/Mc_kg)

    return np.array(times), np.array(GCRs)

# ============================================================
# Run and plot
# ============================================================
def run_planet_model(Mc_me=5.0, a_AU=0.1, t_disk_Myr=3.0, t_end_Gyr=5.0,
                     Z=0.02, dusty=True, init=0.05, mu=2.2, eta=0.1, Lxuv0=3e22,
                     t_sat_Myr=100, decay_index=1.1):

    times, GCRs = evolve_atmosphere(Mc_me, a_AU, t_disk_Myr, t_end_Gyr,
                                    Z, init, dusty, mu, eta,
                                    Lxuv0, t_sat_Myr, decay_index)

    plt.figure(figsize=(7,4))
    plt.loglog(times/1e6, GCRs)
    plt.xlabel("Time (Myr)")
    plt.ylabel("Atmosphere mass fraction $M_{atm}/M_c$")
    plt.title(f"{Mc_me} M⊕ planet at {a_AU} AU")
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.show()

    print(f"Final GCR at {t_end_Gyr} Gyr: {GCRs[-1]:.3e}")
    return times, GCRs

# ============================================================
# Call to run
# ============================================================
if __name__ == "__main__":
    run_planet_model(
        Mc_me=5.0,
        a_AU=0.5,
        t_disk_Myr=3.0,
        t_end_Gyr=5.0,
        init=0.05,
        dusty=True,
        eta=0.1,
        Lxuv0=3e22,
        t_sat_Myr=100,
        decay_index=1.1
    )

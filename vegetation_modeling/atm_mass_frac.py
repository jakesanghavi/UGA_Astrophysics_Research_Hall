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
def cooling_luminosity(Matm, Mc_kg, Rc, Rp, use_Rp_binding=True):
    """
    Return (L, t_KH). Use Rp for binding if use_Rp_binding True.
    """
    # binding energy ~ G * Mc * Matm / R_char
    R_char = Rp if use_Rp_binding else Rc
    E_bind = G * Mc_kg * Matm / max(R_char, 1e-9)

    GCR = Matm / Mc_kg
    # physically: t_KH ~ 1e7 yr * (GCR/0.01)^-3
    # but avoid dividing by extremely tiny GCR that produces insane numbers:
    GCR_eff = max(GCR, 1e-8)   # allow smaller floor than 1e-4 for gradual evolution
    t_KH = 1e7 * year * (GCR_eff / 0.01)**(-3)

    L = E_bind / t_KH
    return L, t_KH

# ============================================================
# Core-powered mass loss
# ============================================================
def core_powered_loss(L, g_surf, Rp):
    denom = max(g_surf * Rp, 1e-20)
    return L / denom

# ============================================================
# Bondi-limited timescale
# ============================================================
def bondi_timescale(Matm, Mc_kg, Rp, T, mu=2.2):
    """
    Better behaved Bondi-limited timescale estimate.
    Uses a surface-shell density approximation:
      rho_surf ~ Matm / (4*pi*Rp^2*H)
    where H = kT/(mu m_p g_surf).
    Returns timescale in seconds (Matm / mdot_B).
    """
    # sound speed
    cs = np.sqrt(k * T / (mu * m_p))

    # Bondi radius (using core mass only; could use total mass)
    R_B = G * Mc_kg / (cs**2 + 1e-12)

    # surface gravity using total mass (core + atm)
    g_surf = G * (Mc_kg + Matm) / max(Rp**2, 1e-12)

    # scale height (avoid zero/Inf)
    H = k * T / (mu * m_p * max(g_surf, 1e-12))
    H = max(H, 1e-6)

    # Assume atmosphere mass is distributed over a shell of thickness DeltaR.
    # Prevent extremely thin shells by forcing a minimum fraction of Rp.
    DeltaR = max(H, 0.05 * Rp)   # treat envelope as at least 5% of Rp thick

    # Conservative surface density estimate (mass spread over shell of area 4πRp^2 and thickness DeltaR)
    rho_surf = Matm / (4.0 * np.pi * Rp**2 * DeltaR)

    # Cap the surface density to avoid absurd values (photosphere-like upper bound)
    # Typical photospheric densities for H2 envelopes are << 1e3 kg/m^3; using 1e-2...1e1 range is conservative.
    rho_cap = 1e2   # kg/m^3  (conservative physical cap; you may lower to 1.0 or 1e-2)
    rho_surf = max(min(rho_surf, rho_cap), 1e-20)

    # Bondi mass-flux estimate (supply rate)
    mdot_B = 4.0 * np.pi * R_B**2 * rho_surf * cs

    # Avoid divide-by-zero
    return Matm / max(mdot_B, 1e-20)

# conservative bondi that also returns mdot_B (kg/s)
def bondi_timescale_and_mdot(Matm, Mc_kg, Rp, T, mu=2.2,
                             rho_cap=1.0, Rb_cap_factor=5.0):
    cs = np.sqrt(k * T / (mu * m_p))
    R_B = G * Mc_kg / max(cs**2, 1e-12)

    g_surf = G * (Mc_kg + Matm) / max(Rp**2, 1e-12)
    H = k * T / (mu * m_p * max(g_surf, 1e-12))
    H = max(H, 1e-6)
    DeltaR = max(H, 0.05 * Rp)

    rho_surf = Matm / (4.0 * np.pi * Rp**2 * DeltaR)
    rho_surf = max(min(rho_surf, rho_cap), 1e-20)

    R_eff = min(R_B, Rb_cap_factor * Rp)
    mdot_B = 4.0 * np.pi * (R_eff**2) * rho_surf * cs

    t_B = Matm / max(mdot_B, 1e-20)
    return t_B, mdot_B, R_B, R_eff, rho_surf


# ============================================================
# XUV mass loss
# ============================================================
def photoevaporation_rate(Rp, M_p, a_AU, L_XUV, eta=0.1):
    d = a_AU * AU
    F_xuv = L_XUV / (4 * pi * d**2)
    q = M_p / M_sun
    R_L = d * 0.49 * q**(1/3)
    xi = max(Rp / max(R_L, 1e-12), 1e-12)
    K = roche_factor(xi)
    return eta * pi * Rp**2 * F_xuv / (G * M_p / Rp) / K

# ============================================================
# Planet radius
# ============================================================
def planet_radius(Rc, Mc_kg, Matm):
    f = Matm/Mc_kg
    C = 0.90 
    beta = 0.26
    
    Rp = Rc * (1 + C * f**beta)
    
    return Rp

# ============================================================
# Atmosphere evolution
# ============================================================
def evolve_atmosphere(Mc_me, a_AU=0.1, t_disk_Myr=3.0, t_end_Gyr=5.0,
                      dusty=True, init=0.05, mu=2.2, eta=0.1,
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
        g_surf = G * (Mc_kg + Matm) / Rp**2

        # Cooling and Bondi times
        L, t_cool = cooling_luminosity(Matm, Mc_kg, Rc, Rp)
        t_B, mdot_B, R_B, R_eff, rho_surf = bondi_timescale_and_mdot(Matm, Mc_kg, Rp, T_eq, mu)

        # compute cooling luminosity (make sure you use Rp and g_surf)
        g_surf = G * (Mc_kg + Matm) / Rp**2
        L, t_cool = cooling_luminosity(Matm, Mc_kg, Rc, Rp)

        # core-powered mass loss (energy-limited estimate)
        mdot_core_energy = core_powered_loss(L, g_surf, Rp)   # = L/(g_surf*Rp)

        # actual core-powered mass loss is limited by Bondi supply:
        mdot_core = min(mdot_core_energy, mdot_B)
        
        L_XUV = Lxuv_track(t, Lxuv0, t_sat_Myr*1e6*year, decay_index)

        # XUV-driven loss as before
        mdot_xuv = photoevaporation_rate(Rp, Mc_kg+Matm, a_AU, L_XUV, eta)

        mdot = mdot_core + mdot_xuv

        dt = min(1e6*year, max(1e3*year, 0.001*Matm/max(mdot,1e-30)))
        Matm = max(Matm - mdot*dt, 0)
        t += dt

        # if show_progress and len(times) % 1 == 0 and t < (t + 1):  # print every step while debugging
        #     print(f"t={t/year:.3e} yr | GCR={Matm/Mc_kg:.3e} | L={L:.3e} W | t_cool={t_cool/year:.3e} yr | t_B={t_B/year:.3e} yr")
        #     print(f"  mdot_core={mdot_core:.3e} kg/s | mdot_xuv={mdot_xuv:.3e} kg/s | mdot_total={mdot:.3e} kg/s | dt={dt/year:.3e} yr")

        if show_progress:
            print(f"t={t/year:.3e} yr | GCR={Matm/Mc_kg:.3e} | L={L:.3e} W | t_cool={t_cool/year:.3e} yr | t_B={t_B/year:.3e} yr")
            print(f"  mdot_core_energy={mdot_core_energy:.3e} kg/s | mdot_B={mdot_B:.3e} kg/s | mdot_core={mdot_core:.3e} kg/s")
            print(f"  mdot_xuv={mdot_xuv:.3e} kg/s | mdot_total={mdot:.3e} kg/s | dt={dt/year:.3e} yr")
            print(f"  Rp={Rp:.3e} m | Rc={Rc:.3e} m | R_B={R_B:.3e} m | R_eff={R_eff:.3e} m | rho_surf={rho_surf:.3e} kg/m3")


        times.append(t/year)
        GCRs.append(Matm/Mc_kg)

    return np.array(times), np.array(GCRs)

# ============================================================
# Run and plot
# ============================================================
def run_planet_model(Mc_me=5.0, a_AU=0.1, t_disk_Myr=3.0, t_end_Gyr=5.0,
                     dusty=True, init=0.05, mu=2.2, eta=0.1, Lxuv0=3e22,
                     t_sat_Myr=100, decay_index=1.1):

    times, GCRs = evolve_atmosphere(Mc_me, a_AU, t_disk_Myr, t_end_Gyr,
                                    init, dusty, mu, eta,
                                    Lxuv0, t_sat_Myr, decay_index)

    plt.figure(figsize=(7,4))
    plt.loglog(times/1e6, GCRs*100)
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
        Mc_me=1.0,
        a_AU=1,
        t_disk_Myr=3.0,
        t_end_Gyr=5.0,
        init=0.05,
        dusty=True,
        eta=0.1,
        Lxuv0=2.05e22,
        t_sat_Myr=100,
        decay_index=1.1
    )

import numpy as np
from atm_mass_frac import evolve_atmosphere, core_radius_zeng, planet_radius, cooling_luminosity
# assume you exposed a version of evolve_atmosphere that can accept show_progress and small dt

Mc_array = [1.0, 2.0, 5.0, 10.0]   # Mearth
a_array  = [0.1, 0.5, 1.0, 5.0, 10.0]

# helper to run one short evolution and return summary numbers (first step)
def one_step_summary(Mc_me, a_AU):
    times, GCRs = evolve_atmosphere(
        Mc_me=Mc_me, a_AU=a_AU, t_disk_Myr=3.0, t_end_Gyr=1e-6,  # tiny t_end -> run only a few steps
        init=0.05, dusty=True, mu=2.2, eta=0.1, Lxuv0=3e22,
        t_sat_Myr=100, decay_index=1.1, show_progress=False
    )
    # evolve_atmosphere should return arrays where first step is after initial boiloff; 
    # if you adapted the evolve function to return diagnostics we can use those too.
    # Instead, it's simpler to call the core functions once to reproduce key numbers
    Mc_kg = Mc_me * 5.9722e24
    Rc = core_radius_zeng(Mc_me)
    # reconstruct Matm from returned GCRs[0]
    GCR0 = GCRs[0]
    Matm = GCR0 * Mc_kg
    Rp = planet_radius(Rc, Mc_kg, Matm)
    # compute T_eq inside your evolve function or recompute:
    from scipy.constants import sigma
    AU = 1.496e11
    L_sun = 3.828e26
    T_eq = (L_sun / (16 * np.pi * (a_AU*AU)**2 * sigma))**0.25

    # compute L, t_cool
    L, t_cool = cooling_luminosity(Matm, Mc_kg, Rc, Rp)

    # get Bondi stuff (call your bondi function if exported; otherwise reimplement minimal)
    from atm_mass_frac import bondi_timescale_and_mdot
    t_B, mdot_B, R_B, R_eff, rho_surf = bondi_timescale_and_mdot(Matm, Mc_kg, Rp, T_eq, mu=2.2)

    # core-powered energy-limited
    g_surf = 6.67430e-11 * (Mc_kg + Matm) / Rp**2
    mdot_core_energy = L / max(g_surf * Rp, 1e-20)

    # XUV
    from atm_mass_frac import photoevaporation_rate
    mdot_xuv = photoevaporation_rate(Rp, Mc_kg+Matm, a_AU, L_XUV=2.05e22, eta=0.1)
    print(mdot_xuv)

    return dict(
        Mc=Mc_me, a=a_AU, T_eq=T_eq, Rp=Rp, Rc=Rc, GCR=GCR0, Matm=Matm,
        L=L, t_cool=t_cool, t_B=t_B, mdot_B=mdot_B, mdot_core_energy=mdot_core_energy,
        mdot_xuv=mdot_xuv, g_surf=g_surf, R_B=R_B, R_eff=R_eff, rho_surf=rho_surf
    )

# run grid and print table
rows=[]
for Mc in Mc_array:
    for a in a_array:
        s = one_step_summary(Mc, a)
        rows.append(s)
        print(f"Mc={s['Mc']:4.1f} M⊕ | a={s['a']:4.2f} AU | T_eq={s['T_eq']:.0f} K | Rp={s['Rp']/6.371e6:.3f} R⊕")
        print(f"  GCR={s['GCR']:.3e} | Matm={s['Matm']:.3e} kg")
        print(f"  L={s['L']:.3e} W | t_cool={s['t_cool']/ (365.25*24*3600):.3e} yr")
        print(f"  t_B={s['t_B']/ (365.25*24*3600):.3e} yr | mdot_B={s['mdot_B']:.3e} kg/s")
        print(f"  mdot_core_energy={s['mdot_core_energy']:.3e} kg/s | mdot_xuv={s['mdot_xuv']:.3e} kg/s")
        print("-"*80)

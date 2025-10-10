# Atmosphere Evolution Multi-Run Notebook
# ============================================================
# Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext
from atm_mass_frac import evolve_atmosphere

# Masses and orbital distances
Mc_array = [1, 2, 5, 10]           # Earth masses
a_array  = [0.1, 0.5, 1, 2, 5, 10, 25]  # AU
eta = 0.1

# Create 2x2 subplot grid
fig, axes = plt.subplots(2, 2, figsize=(12,8))
axes = axes.flatten()

for i, Mc in enumerate(Mc_array):
    ax = axes[i]
    for a in a_array:
        times, GCRs = evolve_atmosphere(
            Mc_me=Mc,
            a_AU=a,
            t_disk_Myr=3.0,
            t_end_Gyr=5.0,
            dusty=True,
            eta=eta,
            Lxuv0=3e22,
            t_sat_Myr=100,
            decay_index=1.1
        )
        ax.loglog(times/1e6, GCRs, label=f"a={a} AU")
    
    # Formatting subplot
    ax.set_title(f"{Mc} M⊕ planet")
    ax.set_xlabel("Time (Myr)")
    ax.set_ylabel("Atmosphere mass fraction $M_{{atm}}/M_c$")
    
    # Log ticks
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.grid(True, which="both", ls=":")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.show()
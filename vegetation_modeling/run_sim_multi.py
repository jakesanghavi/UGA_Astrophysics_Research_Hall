import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext
from atm_mass_frac import evolve_atmosphere

# Param grid
Mc_array = [1, 2, 5, 10]
a_array  = [0.1, 0.5, 1, 2, 5, 10, 25]
# Mc_array = [1]
# a_array = [1]
eta = 0.3
init = 0.15

# 2x2
fig, axes = plt.subplots(2, 2, figsize=(12,8))
axes = axes.flatten()

all_times = []
all_GCRs = []

for Mc in Mc_array:
    for a in a_array:
        times, GCRs = evolve_atmosphere(
            Mc_me=Mc,
            a_AU=a,
            t_disk_Myr=3.0,
            t_end_Gyr=5.0,
            init=init,
            dusty=True,
            eta=eta,
            Lxuv0=1.75e22,
            t_sat_Myr=100,
            decay_index=1.1
        )
        all_times.append(times/1e6)
        all_GCRs.append(GCRs)

# Global things for same limits
x_min = min([np.min(t) for t in all_times])
x_max = max([np.max(t) for t in all_times])
y_min = min([np.min(g) for g in all_GCRs])
y_max = max([np.max(g) for g in all_GCRs])

for i, Mc in enumerate(Mc_array):
    ax = axes[i]
    for j, a in enumerate(a_array):
        times, GCRs = evolve_atmosphere(
            Mc_me=Mc,
            a_AU=a,
            t_disk_Myr=3.0,
            t_end_Gyr=5.0,
            init=init,
            dusty=True,
            eta=eta,
            Lxuv0=2.05e22,
            t_sat_Myr=100,
            decay_index=1.1
        )
        ax.loglog(times/1e6, GCRs, label=f"a={a} AU")
    
    ax.set_title(f"{Mc} M⊕ planet")
    ax.set_xlabel("Time (Myr)")
    ax.set_ylabel(r"Atmosphere mass fraction $M_{\rm atm}/M_c$")
        
    # Set same scale for all subplots
    ax.set_xlim(x_min, x_max)
    
    # ylim very annoying
    # y_min_pow10 = 10**np.floor(np.log10(y_min))
    # y_max_pow10 = 10**np.ceil(np.log10(y_max))
    y_min_pow10 = 10**(-7)
    y_max_pow10 = 10 ** (-0.5)
    ax.set_ylim(y_min_pow10, y_max_pow10)
    
    # Major ticks at powers of 10 only
    ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=None, numticks=100))
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=None, numticks=100))

    # No minor ticks(not working)
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=[], numticks=100))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=[], numticks=100))
    ax.legend(fontsize=8)

plt.suptitle(fr"Atmospheric Mass Ratio - Initial $H_2$ Concentration = {init}")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f"{init}.png")
plt.show()

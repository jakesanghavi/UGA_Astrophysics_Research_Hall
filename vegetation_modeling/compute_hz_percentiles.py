import numpy as np
import math

def calc_hz_percentiles(m_target):
    Lsol = 3.828e26

    # Kopparapu coefficients
    seffsun  = [1.776,1.107, 0.356, 0.320, 1.188, 0.99]
    a = [2.136e-4, 1.332e-4, 6.171e-5, 5.547e-5, 1.433e-4, 1.209e-4]
    b = [2.533e-8, 1.580e-8, 1.698e-9, 1.526e-9, 1.707e-8, 1.404e-8]
    c = [-1.332e-11, -8.308e-12, -3.198e-12, -2.874e-12, -8.968e-12, -7.418e-12]
    d = [-3.097e-15, -1.931e-15, -5.575e-16, -5.011e-16, -2.084e-15, -1.713e-15]

    RUNAWAY = 1
    MAXIMUM = 2

    # Load Seiss isochrone
    isochrone_file = '5interpolated_seiss_1E9.dat'

    mstar, tstar, lstar = [], [], []

    with open(isochrone_file) as f:
        for line in f:
            cols = line.strip().split()
            mstar.append(float(cols[3]))
            tstar.append(float(cols[2]))
            lstar.append(float(cols[0]) * Lsol)

    mstar = np.array(mstar)
    tstar = np.array(tstar)
    lstar = np.array(lstar)

    # Compute HZ boundaries for all models
    runaway_greenhouse, maximum_greenhouse = [], []

    for i in range(len(mstar)):
        Tstar = tstar[i] - 5780.0

        def seff(idx):
            return (
                seffsun[idx]
                + a[idx]*Tstar
                + b[idx]*Tstar**2
                + c[idx]*Tstar**3
                + d[idx]*Tstar**4
            )

        S_run = seff(RUNAWAY)
        S_max = seff(MAXIMUM)

        d_inner = math.sqrt((lstar[i]/Lsol) / S_run)
        d_outer = math.sqrt((lstar[i]/Lsol) / S_max)

        runaway_greenhouse.append(d_inner)
        maximum_greenhouse.append(d_outer)

    runaway_greenhouse = np.array(runaway_greenhouse)
    maximum_greenhouse = np.array(maximum_greenhouse)

    # Find closest model to target stellar mass
    idx = np.argmin(np.abs(mstar - m_target))
    d_inner = runaway_greenhouse[idx]
    d_outer = maximum_greenhouse[idx]

    # Compute percentiles (linear interpolation)
    percentiles = [10, 50, 90]
    p_vals = [d_inner + (p/100)*(d_outer - d_inner) for p in percentiles]
    # # Log space percentiles if better
    # p_vals = [
    #     10**(
    #         np.log10(d_inner)
    #         + (p/100.0)*(np.log10(d_outer) - np.log10(d_inner))
    #     )
    #     for p in percentiles
    # ]

    return p_vals

print(calc_hz_percentiles(1.0))
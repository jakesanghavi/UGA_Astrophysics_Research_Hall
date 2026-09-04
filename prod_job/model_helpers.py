import exoplasim as exo
import numpy as np
from matplotlib import pyplot as plt
from veg_utils import calc_f_ret_big_rcb, calc_r_c, calc_r_B, calc_rho_rcb, calc_m_atm, calc_r_prime_b, calc_hz_percentiles
from constants import mearth, Gsi, pi, rearth, lsol, rsol, sigma_SB
from atm_mass_frac import evolve_atmosphere
import sys
import json
import shutil
import os
import subprocess
import base64
import pickle
import uuid
from scipy.interpolate import interp1d
from concurrent.futures import ProcessPoolExecutor, as_completed

### PLANET CONFIGURATION ###
N_YEARS = 5
X_fe = 0.32
X_fe_m=0.081
RESOLUTION = 'T21'
# For some reason N=6 crashes everything
# NCPUS (MPI ranks per model) and WORKERS (planets run concurrently) can be set
# from the environment so a batch script can size them to the SLURM allocation.
# Keep WORKERS * NCPUS <= available cores. The defaults suit a local machine.
NCPUS = int(os.environ.get("EXOPLASIM_NCPUS", "4"))
# Each worker launches its own ExoPlaSim process using NCPUS MPI ranks.
# WORKERS = 1 reproduces the original fully sequential behavior.
WORKERS = int(os.environ.get("EXOPLASIM_WORKERS", "2"))
# On a crash, retry the run this many times before recording the point as
# crashed. Each retry uses a fresh seed AND a smaller timestep (see below).
MAX_RETRIES = 2

# Model timestep in minutes. ExoPlaSim's T21 default is 45 min, tuned for
# Earth's gravity. Low-gravity planets have a larger atmospheric scale height
# and need a finer step for numerical stability, so calculate_veg scales this
# down with surface gravity (and shrinks it further on each retry), floored at
# MIN_TIMESTEP_MIN so runs don't become absurdly slow.
BASE_TIMESTEP_MIN = 45.0
MIN_TIMESTEP_MIN = 3.0
G_EARTH = 9.80665

NLAYERS = 10
PRECISION = 4
OUTPUT_TYPE = '.nc'
PLANET_NAME = 'EARTH'
# MPs = [0.1, 0.15, 0.25, 0.5, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.00]

# Vegetation settings
VEGETATION = 2
VEGACCEL = 1
INIT_GROWTH = 0.5
WET_SOIL = True
BASE_FLUX = 1367

# Planet Comparison to Earth
PRESSURE_FRACTION = 1
MASS_RATIO=3
#MSTARS = [0.1, 0.5, 1, 1.25, 1.5, 2, 5, 9]
MSTARS = [0.7, 0.8, 0.9, 1.0, 1.1]
#MSTARS = [0.1, 0.5, 1, 1.25, 1.5, 2]

# Gas settings
F_INIT = 0.01

os.environ["GFORTRAN_ERROR_BACKTRACE"] = "1"
os.environ["GFORTRAN_UNBUFFERED_ALL"] = "1"
os.environ["FORTRAN_STDOUT_UNIT"] = "6"
os.environ["PYTHONFAULTHANDLER"] = "1"

def get_lxuv0_from_bolometric(L_bol_present):
    L_bol_watts = L_bol_present

    f_sat = 10**(-3.02)
    Lxuv0_watts = L_bol_watts * f_sat
    
    return Lxuv0_watts

# Estimate radius of the planet based on its mass
# This is based on "The mass–radius relation of exoplanets revisited" by Müller et al. 2024
def piecewise_radius_estimate(mass_ratio):
    # Small/rocky planets, like Earth
    if mass_ratio < 4.37:
        return 1.02 * (mass_ratio**0.27)
    # Intermediate-mass planets
    # H/He envelopes no longer neglible, so radius grows faster with mass than before
    if mass_ratio < 127:
        return 0.56 * (mass_ratio**0.67)
    # Massive planets, mass dominated by light gas.
    # Radius becomes almost constant and independent of mass
    # This gas is semi-degenerate, leading to the constant relation
    return 18.6 * (mass_ratio ** (-0.06))

def radius_noack(m_p):
    return 1000*((7030 - 1840 * X_fe)*(m_p)**0.282)/rearth

def cmf_noack():
    return (X_fe - X_fe_m)/(1-X_fe_m)

def core_radius_noack(m_p, cmf):
    return 1000*(4850*(cmf**0.328)*(m_p)**0.266)

def core_density_noack(m_p, cmf, r_c):
    return (cmf*m_p*mearth)/(4/3 * np.pi * ((r_c)**3))

def core_mass_noack(r_c, rho_c):
    return (4/3)*np.pi*(r_c**3)*rho_c

def try_run_planet(planet, years):
    # Serialize planet into base64 string for passing via stdin
    data = pickle.dumps(planet)
    b64 = base64.b64encode(data).decode()

    # Run all requested years inside a single isolated subprocess instead of
    # spawning one interpreter per year. Postprocessing is skipped for every
    # year except the last: the caller only inspects the final year, so this
    # avoids redundant pyburn passes while producing bit-identical physics and
    # final-year output.
    code = f"""
import pickle, base64, exoplasim

planet = pickle.loads(base64.b64decode(input()))

n = {int(years)}
if n > 1:
    planet.run(years=n - 1, postprocess=False, clean=False)
planet.run(years=1, postprocess=True, clean=False)

print(base64.b64encode(pickle.dumps(planet)).decode())
"""

    proc = subprocess.run(
        [sys.executable, "-c", code],
        input=b64.encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Crash case (SIGILL, segfault, etc)
    if proc.returncode != 0:
        print(f"[try_run_planet] subprocess exited with code {proc.returncode}")
        if proc.stderr:
            print("STDERR (tail):\n", proc.stderr.decode()[-2000:])
        return 0

    out_b64 = proc.stdout.decode().strip()
    return pickle.loads(base64.b64decode(out_b64))


planet_params = {
        'vegetation': VEGETATION,
        'vegaccel': VEGACCEL,
        'initgrowth': INIT_GROWTH,
        'wetsoil': WET_SOIL,
        'pH2': 0.0,
        'pHe': 5.24e-6,
        'pN2': 0.78084,
        'pO2': 0.20946,
        'pCO2': 330.0e-6,
        'pAr': 9.34e-3,
        'pNe': 18.18e-6,
        'pKr': 1.14e-6,
        'pH2O': 0.01,
        'pCH4': 0.0
    }

gas_params = ['pH2', 'pHe', 'pN2', 'pO2', 'pCO2', 'pAr', 'pNe', 'pKr', 'pH2O', 'pCH4']

for param in gas_params:
    if param in planet_params:
        planet_params[param] *= PRESSURE_FRACTION

mass_grid = np.array([
    0.010, 0.015, 0.020, 0.030, 0.040, 0.050, 0.060,
    0.070, 0.072, 0.075, 0.080, 0.090, 0.100, 0.110,
    0.130, 0.150, 0.170, 0.200, 0.300, 0.400, 0.500,
    0.600, 0.700, 0.800, 0.900, 1.000, 1.100, 1.200,
    1.300, 1.400
])

teff_grid = np.array([
    2142., 2406., 2545., 2706., 2784., 2831., 2870.,
    2906., 2911., 2919., 2931., 2956., 2998., 3016.,
    3052., 3088., 3127., 3196., 3404., 3582., 3731.,
    3868., 3998., 4120., 4232., 4330., 4421., 4505.,
    4588., 4671.
])

log_L_grid = np.array([
    -3.13, -2.79, -2.57, -2.17, -1.92, -1.74, -1.61,
    -1.51, -1.49, -1.46, -1.42, -1.36, -1.44, -1.39,
    -1.32, -1.25, -1.18, -1.10, -0.86, -0.68, -0.54,
    -0.42, -0.31, -0.21, -0.13, -0.05,  0.02,  0.09,
     0.15,  0.22
])

get_teff = interp1d(
    mass_grid, teff_grid,
    kind='linear',
    bounds_error=False,
    fill_value='extrapolate'
)

get_logL = interp1d(
    mass_grid, log_L_grid,
    kind='linear',
    bounds_error=False,
    fill_value='extrapolate'
)

mass_grid_2_gyr = np.array([
    0.070, 0.072, 0.075, 0.080, 0.090, 0.100, 0.110, 0.130, 0.150, 0.170,
    0.200, 0.300, 0.400, 0.500, 0.600, 0.700, 0.800, 0.900, 1.000, 1.100, 1.200
])

teff_grid_2_gyr = np.array([
    1631., 1790., 2041., 2343., 2644., 2811., 2916., 3046., 3131., 3192.,
    3262., 3416., 3520., 3680., 3979., 4418., 4878., 5297., 5697., 5996., 6053.
])

# Logarithmic luminosities log(L/L_sun) at 2 Gyr
log_l_grid_2gyr = np.array([
    -4.22, -4.05, -3.81, -3.52, -3.21, -3.02, -2.88, -2.69, -2.54, -2.42,
    -2.28, -1.94, -1.68, -1.42, -1.13, -0.83, -0.54, -0.27, -0.01,  0.25,  0.45
])

teff_grid_4_gyr = np.array([
    1631., 1790., 2041., 2343., 2644., 2811., 2916., 3046., 3131., 3192.,
    3262., 3416., 3520., 3680., 3979., 4418., 4878., 5297., 5697., 5996., 6053.
])

# Logarithmic luminosities log(L/L_sun) at 4 Gyr

log_l_grid_4gyr = np.array([
    -4.29, -4.12, -3.88, -3.58, -3.25, -3.06, -2.92, -2.73, -2.58, -2.46,
    -2.32, -1.97, -1.71, -1.45, -1.15, -0.85, -0.56, -0.29, -0.03,  0.23,  0.43
])

get_2gyr_logL = interp1d(mass_grid_2_gyr, log_l_grid_4gyr, kind='linear', bounds_error=False, fill_value="extrapolate")
get_2gyr_teff = interp1d(mass_grid_2_gyr, teff_grid_4_gyr, kind='linear', bounds_error=False, fill_value="extrapolate")

def stellar_mass_to_temp_flux(M_star, a):
    startemp = float(get_2gyr_teff(M_star))
    
    log_L = float(get_2gyr_logL(M_star))
    L_star = (10**log_L) * lsol

    a_m = a * 1.496e11

    flux = L_star / (4 * np.pi * a_m**2)

    return startemp, flux

def calculate_veg(mass_ratio, mstar, au, resolution, to_append):
    r_new = radius_noack(mass_ratio)
    g_new = 9.80665 * mass_ratio / (r_new ** 2)
    startemp, flux = stellar_mass_to_temp_flux(mstar, au)
    
    # Cap flux as it crashes model at low AU
    # flux = min(flux, 2000)

    # Work on a per-call copy so concurrent grid points never clobber each
    # other's configuration (and so results are identical to the sequential run).
    local_params = dict(planet_params)

    local_params['gravity'] = g_new
    local_params['radius'] = r_new
    # local_params['flux'] = BASE_FLUX / (au**2)
    local_params['startemp'] = startemp
    local_params['flux'] = flux
    
    CMF = cmf_noack()
    r_c = core_radius_noack(mass_ratio, CMF)
    rhoc = core_density_noack(mass_ratio, CMF, r_c)
    m_c = core_mass_noack(r_c, rhoc)/mass_ratio
    r_rcb = 2 * r_c
    t_eq = 255
    r_b = calc_r_B(m_c, t_eq)
    rho_rcb = calc_rho_rcb(r_b, r_rcb)
    r_prime_b = calc_r_prime_b(r_b)
    log_L = float(get_logL(mstar))
    L_star = (10**log_L) * lsol
    Lxuv0 = get_lxuv0_from_bolometric(L_star)

    retained_frac = np.clip(calc_f_ret_big_rcb(m_c, t_eq, r_c, r_rcb), 0, 0.5)
    times, GCRs, xuvs = evolve_atmosphere(
                M_p=mass_ratio,
                a_AU=au,
                t_disk_Myr=3.0,
                t_end_Gyr=5.0,
                init=F_INIT,
                dusty=True,
                eta=0.1,
                Lnow=L_star,
                t_sat_Myr=100,
                decay_index=1.1,
                M_star = mstar
            )

    # Targeting a time of 4.5 Gyr
    target_time = 4.5 * 10**9
    mask = times > target_time
    target_index = np.argmax(mask)
    F = GCRs[target_index]
    
    # If time is greater than max time, that means there has been no gas retained
    # as the simulation stops when M_atm == 0
    if target_index <= 0:
        F = 0
    local_params['pHe'] = 0.25 * Gsi * F * retained_frac * (mass_ratio * mearth) ** 2 * 10 ** (-10)  / (4 * pi * (r_new * rearth) ** 4)
    local_params['pH2'] = 0.75 * Gsi * F * retained_frac * (mass_ratio * mearth) ** 2 *  10 ** (-10) / (4 * pi * (r_new * rearth) ** 4)

    # When running grid points concurrently, each planet launches its own
    # mpiexec. Without this, every mpiexec binds its ranks starting at core 0,
    # so concurrent runs fight over the same cores. "--bind-to none" lets the OS
    # scheduler spread them across all cores. Single-run behavior is unchanged.
    mpi_opts = "--bind-to none" if WORKERS > 1 else None

    # Low-gravity planets have a larger atmospheric scale height and need a finer
    # timestep to stay numerically stable, so scale the Earth-tuned default down
    # with surface gravity (capped at the default, floored so runs stay tractable).
    base_timestep = max(MIN_TIMESTEP_MIN, BASE_TIMESTEP_MIN * min(1.0, g_new / G_EARTH))

    # Low-gravity planets (especially at high insolation) can still hit an
    # intermittent, weather- (and therefore seed-) dependent numerical
    # instability in PlaSim's surface-flux scheme ("negative z/z0"). Retry a
    # crashed run up to MAX_RETRIES times, each time with a fresh random seed AND
    # a smaller timestep. If every attempt crashes, return [None, None] so the
    # caller can tell a genuine crash apart from a real zero-vegetation result.
    planet = None
    for attempt in range(MAX_RETRIES + 1):
        # Halve the timestep on each retry (down to the floor); a finer step is a
        # more deterministic stability fix than reseeding alone.
        local_params['timestep'] = max(MIN_TIMESTEP_MIN, base_timestep / (2 ** attempt))
        try:
            shutil.rmtree(f"custom_earthlike_model{to_append}")
            shutil.rmtree(f"custom_earthlike_model_crashed{to_append}")
        except:
            pass

        planet = exo.Model(
            workdir=f"custom_earthlike_model{to_append}",
            modelname=f"custom_earthlike_model{to_append}",
            resolution=resolution,
            ncpus=NCPUS,
            layers=NLAYERS,
            precision=PRECISION,
            outputtype=OUTPUT_TYPE,
            mpi_opts=mpi_opts
        )

        planet.debug = True
        planet.verbose = True

        planet.configure(**local_params)
        # Distinct nonzero seed per attempt, from OS entropy so it is independent
        # across worker processes. PlaSim seeds its initial white-noise
        # perturbation from this instead of the wall clock.
        seed = int.from_bytes(os.urandom(4), "little") % (2 ** 31 - 1) or 1
        planet._edit_namelist("plasim_namelist", "SEED", str(seed))
        planet.exportcfg()

        planet = try_run_planet(planet, N_YEARS)

        if planet != 0 and planet is not None:
            break  # success

        if attempt < MAX_RETRIES:
            next_dt = max(MIN_TIMESTEP_MIN, base_timestep / (2 ** (attempt + 1)))
            print(f"[calculate_veg] run crashed (mass={mass_ratio}, mstar={mstar}, "
                  f"au={au}); retrying with a new seed and timestep={next_dt:.1f} min "
                  f"({attempt + 1}/{MAX_RETRIES})")

    if planet == 0 or planet is None:
        # Every attempt crashed: distinct from a genuine zero-vegetation result.
        return [None, None]

    veg = planet.inspect("veggpp", tavg=True)
    land = planet.inspect('lsm')
    land = np.sum(land, axis=0)
    
    land_mask = land > 0
    masked_veg_values = veg[land_mask]
    average_veg = np.mean(masked_veg_values)
    tot_veg = np.sum(masked_veg_values)
        
    return [average_veg, tot_veg]

def _grid_point(mass_ratio, mstar, au, resolution, workdir_suffix):
    """Compute a single grid point. Runs inside a worker process for isolation
    (ExoPlaSim uses process-wide os.chdir, so concurrency must be process-based)."""
    veg_amt = calculate_veg(mass_ratio, mstar, au, resolution, workdir_suffix)
    flux = stellar_mass_to_temp_flux(mstar, au)
    veg_amt.append(flux[0])
    veg_amt.append(flux[1])
    # A crashed run yields None for the vegetation entries (kept as JSON null),
    # which is distinct from a genuine 0.0 vegetation result.
    return [float(v) if v is not None else None for v in veg_amt]


def model_fun(mass_ratio, resolution="T21", points=None, file_tag="", output_file=None):
    """Evaluate one planet mass over a set of (stellar mass, semi-major axis) points.

    points      : explicit list of (mstar, au) tuples to evaluate. If None
                  (default), sweep every star in MSTARS across its habitable-zone
                  percentiles (the original behavior).
    file_tag    : inserted into the default output filename (e.g. "_massonly")
                  so alternate configurations are easy to tell apart.
    output_file : full override of the output filename (takes precedence over
                  file_tag), used e.g. for the Earth reference baseline.
    """
    # Change name based on resolution
    res_suffix = "" if resolution == 'T21' else resolution
    if output_file is None:
        output_file = f"16cpus_test_{str(mass_ratio).replace('.', '')}{file_tag}{res_suffix}.json"

    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            output_dict = json.load(f)
    else:
        output_dict = {}

    # Which (stellar mass, semi-major axis) points to evaluate for this mass.
    if points is None:
        points = [(mstar, au) for mstar in MSTARS for au in calc_hz_percentiles(mstar)]

    # Collect the grid points that still need to be computed.
    tasks = []
    for mstar, au in points:
        ms = str(mstar)
        au_key = str(au)
        existing = output_dict.get(ms, {}).get(au_key)
        # Skip points already computed successfully; re-attempt ones that
        # previously crashed (vegetation stored as null).
        if existing is not None and existing[0] is not None:
            continue
        # Unique workdir per task so concurrent runs never share a directory.
        safe = f"{ms}_{au_key}".replace('.', 'p').replace('-', 'm')
        workdir_suffix = f"{res_suffix}_{safe}"
        tasks.append((ms, au_key, mstar, au, workdir_suffix))

    if not tasks:
        return

    def _save(ms, au_key, result):
        output_dict.setdefault(ms, {})[au_key] = result
        with open(output_file, "w") as f:
            json.dump(output_dict, f, indent=4)

    # Run the first task on its own first: this compiles the ExoPlaSim binary
    # (a shared, one-time step) before any concurrent workers start, avoiding a
    # compilation race. It is also the whole job when WORKERS == 1.
    ms, au_key, mstar, au, workdir_suffix = tasks[0]
    try:
        _save(ms, au_key, _grid_point(mass_ratio, mstar, au, resolution, workdir_suffix))
    except Exception as e:
        print(f"Error! ({ms}, {au_key}): {e}")

    remaining = tasks[1:]
    if not remaining:
        return

    if WORKERS <= 1:
        for ms, au_key, mstar, au, workdir_suffix in remaining:
            try:
                _save(ms, au_key, _grid_point(mass_ratio, mstar, au, resolution, workdir_suffix))
            except Exception as e:
                print(f"Error! ({ms}, {au_key}): {e}")
        return

    # Independent grid points run concurrently in isolated worker processes.
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        future_map = {
            pool.submit(_grid_point, mass_ratio, mstar, au, resolution, workdir_suffix): (ms, au_key)
            for ms, au_key, mstar, au, workdir_suffix in remaining
        }
        for fut in as_completed(future_map):
            ms, au_key = future_map[fut]
            try:
                _save(ms, au_key, fut.result())
            except Exception as e:
                print(f"Error! ({ms}, {au_key}): {e}")

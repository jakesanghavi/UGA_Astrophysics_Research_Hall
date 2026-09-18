"""SIMBA light-limited GPP factor reconstruction.

Matches `simba.f90` `vegstep` (ExoPlaSim):

    zbeta = max(0, 1 + 0.3 * ln((CO2 - 0) / (360 - 0)))   # 0 if CO2 ~ 0
    zft   = clip((T_sfc_K - 273.15) / 5, 0, 1)
    zveg  = 1 - exp(-0.5 * LAI)   # output LAI is inverted from cover
    zgppl = 3.4e-10 * zbeta * zft * zveg * SW_down
    zgpp  = min(zgppl, zgppw)

The docs write beta as ``1 + max(0, BF ln(...))`` (never below 1). The Fortran
does ``max(0, 1 + BF ln(...))``, so beta *can* drop below 1 and hits 0 near
13 ppm. This module uses the Fortran form.

Annual-mean of a product is not the product of annual means. `summarize_from_fields`
reconstructs per cell (and per time, if present) and reports the relative error
against SIMBA's own `vegppl`.
"""

import math

import numpy as np

from atmos_presets import co2_ppmv_from_params, total_pressure_bar

# SIMBA / landmod constants (mks).
RLUE = 3.4e-10          # kg C / J  (epsilon_luemax)
CO2_REF = 360.0         # ppmv
CO2_COMP = 0.0          # ppmv
CO2_SENS = 0.3          # beta factor BF
T_CRIT_C = 5.0          # deg C
TMELT = 273.15          # K
K_VEG = 0.5             # Beer's-law extinction
CO2_ZERO = 1.0e-8       # Fortran branch: no CO2 => zbeta = 0

LIGHT_FACTORS = ("beta", "fT", "fPAR", "SWdn")


def beta_co2(co2_ppmv):
    """Carbon-dioxide multiplier from the Fortran, not the docs."""
    co2 = float(co2_ppmv)
    if co2 < CO2_ZERO:
        return 0.0
    denom = CO2_REF - CO2_COMP
    if denom <= 0.0:
        return 0.0
    argument = (co2 - CO2_COMP) / denom
    if argument <= 0.0:
        return 0.0
    return float(max(0.0, 1.0 + CO2_SENS * math.log(argument)))


def f_temperature(ts_k):
    """Temperature limitation; `ts_k` in Kelvin, any shape."""
    ts_c = np.asarray(ts_k, dtype=float) - TMELT
    return np.clip(ts_c / T_CRIT_C, 0.0, 1.0)


def f_par(lai):
    """Vegetation cover / fPAR from output LAI: 1 - exp(-k LAI)."""
    lai = np.asarray(lai, dtype=float)
    lai = np.clip(lai, 0.0, None)
    return 1.0 - np.exp(-K_VEG * lai)


def sw_down(rss, ssru=None):
    """Downward surface shortwave.

    SIMBA multiplies by `dfd` (downward SW). Pyburn gives net SW (`rss`) and
    upward SW (`ssru`). If upward is available and same-signed-positive,
    down = net + up. Otherwise fall back to net (a lower bound).
    """
    rss = np.asarray(rss, dtype=float)
    if ssru is None:
        return rss
    ssru = np.asarray(ssru, dtype=float)
    if ssru.shape != rss.shape:
        return rss
    return rss + ssru


def gpp_light(beta, fT, fPAR, SWdn, rlue=RLUE):
    """Reconstructed light-limited GPP (same broadcasting as the inputs)."""
    return rlue * beta * np.asarray(fT) * np.asarray(fPAR) * np.asarray(SWdn)


def land_mask_from_lsm(lsm):
    """True on land. Matches `calculate_veg`: sum any time axis, then > 0."""
    lsm = np.asarray(lsm)
    if lsm.ndim >= 3:
        return np.sum(lsm, axis=0) > 0
    return np.asarray(lsm) > 0


def _as_array(field):
    field = np.asarray(field, dtype=float)
    while field.ndim > 3:
        field = np.squeeze(field)
    return field


def _align_fields(fields):
    """Broadcast-safe: keep a shared 3D shape, otherwise time-mean every 3D field."""
    arrays = {name: _as_array(value) for name, value in fields.items() if value is not None}
    shapes = {arr.shape for arr in arrays.values()}
    if len(shapes) <= 1:
        return arrays
    aligned = {}
    for name, arr in arrays.items():
        if arr.ndim == 3:
            aligned[name] = np.nanmean(arr, axis=0)
        else:
            aligned[name] = arr
    return aligned


def _spatial_mask_mean(field, land_mask):
    """Mean of `field` on land. Accepts (y, x) or (t, y, x)."""
    field = np.asarray(field, dtype=float)
    mask = np.asarray(land_mask, dtype=bool)
    if field.ndim == 3:
        if mask.shape != field.shape[1:]:
            raise ValueError(
                f"land mask {mask.shape} does not match field {field.shape}"
            )
        values = field[:, mask]
    elif field.ndim == 2:
        if mask.shape != field.shape:
            raise ValueError(
                f"land mask {mask.shape} does not match field {field.shape}"
            )
        values = field[mask]
    else:
        raise ValueError(f"expected 2D or 3D field, got {field.shape}")
    if values.size == 0:
        return None
    return float(np.nanmean(values))


def _finite(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def dln(value, reference):
    """ln(value / reference); None if either side is missing or non-positive."""
    value = _finite(value)
    reference = _finite(reference)
    if value is None or reference is None or value <= 0.0 or reference <= 0.0:
        return None
    return float(math.log(value / reference))


def dominant_term(dln_terms):
    """Name of the largest-|dln| entry; None if nothing is comparable."""
    ranked = [
        (abs(delta), name)
        for name, delta in dln_terms.items()
        if delta is not None
    ]
    if not ranked:
        return None
    ranked.sort(reverse=True)
    return ranked[0][1]


def base_record(
    *,
    mass_ratio,
    mstar,
    au,
    atmos_type,
    crashed=False,
    gravity=None,
    radius=None,
    toa_flux=None,
    startemp=None,
    avg_gpp=None,
    tot_gpp=None,
):
    """JSON-safe skeleton for one diagnostic grid point (including crashes)."""
    return {
        "atmos": atmos_type,
        "mass_ratio": _finite(mass_ratio),
        "mstar": _finite(mstar),
        "au": _finite(au),
        "crashed": bool(crashed),
        "gravity": _finite(gravity),
        "radius_rearth": _finite(radius),
        "toa_flux": _finite(toa_flux),
        "startemp": _finite(startemp),
        "avg_gpp": _finite(avg_gpp),
        "tot_gpp": _finite(tot_gpp),
        "p_total_bar": None,
        "pCO2_bar": None,
        "pH2_bar": None,
        "pHe_bar": None,
        "co2_ppmv": None,
        "epsilon": RLUE,
        "beta": None,
        "gpp": None,
        "gppl": None,
        "gppw": None,
        "frac_light_limited": None,
        "fT": None,
        "fPAR": None,
        "lai": None,
        "SWdn": None,
        "ts_C": None,
        "mrso": None,
        "evap": None,
        "gppl_reconstructed": None,
        "gppl_recon_relerr": None,
    }


def summarize_from_fields(
    *,
    gpp,
    gppl,
    gppw,
    lai,
    ts,
    rss,
    land_mask,
    params,
    mass_ratio,
    mstar,
    au,
    atmos_type,
    ssru=None,
    mrso=None,
    evap=None,
    gravity=None,
    radius=None,
    toa_flux=None,
    startemp=None,
    avg_gpp=None,
    tot_gpp=None,
    crashed=False,
):
    """Collapse one run's inspect fields into JSON-safe land-mean diagnostics."""
    record = base_record(
        mass_ratio=mass_ratio,
        mstar=mstar,
        au=au,
        atmos_type=atmos_type,
        crashed=crashed,
        gravity=gravity,
        radius=radius,
        toa_flux=toa_flux,
        startemp=startemp,
        avg_gpp=avg_gpp,
        tot_gpp=tot_gpp,
    )
    if crashed:
        return record

    p_tot = total_pressure_bar(params)
    co2_ppmv = co2_ppmv_from_params(params)
    beta = beta_co2(co2_ppmv)
    record["p_total_bar"] = _finite(p_tot)
    record["pCO2_bar"] = _finite(params.get("pCO2"))
    record["pH2_bar"] = _finite(params.get("pH2"))
    record["pHe_bar"] = _finite(params.get("pHe"))
    record["co2_ppmv"] = _finite(co2_ppmv)
    record["beta"] = _finite(beta)

    aligned = _align_fields({
        "gpp": gpp, "gppl": gppl, "gppw": gppw, "lai": lai, "ts": ts,
        "rss": rss, "ssru": ssru, "mrso": mrso, "evap": evap,
    })
    gpp = aligned["gpp"]
    gppl = aligned["gppl"]
    gppw = aligned["gppw"]
    lai = aligned["lai"]
    ts = aligned["ts"]
    rss = aligned["rss"]
    ssru = aligned.get("ssru")
    mrso = aligned.get("mrso")
    evap = aligned.get("evap")

    fT = f_temperature(ts)
    fPAR = f_par(lai)
    SWdn = sw_down(rss, ssru)
    rec_gppl = gpp_light(beta, fT, fPAR, SWdn)

    gppl_a = np.asarray(gppl, dtype=float)
    gppw_a = np.asarray(gppw, dtype=float)
    light_wins = gppl_a < gppw_a

    record["gpp"] = _spatial_mask_mean(gpp, land_mask)
    record["gppl"] = _spatial_mask_mean(gppl_a, land_mask)
    record["gppw"] = _spatial_mask_mean(gppw_a, land_mask)
    record["frac_light_limited"] = _spatial_mask_mean(
        light_wins.astype(float), land_mask
    )
    record["fT"] = _spatial_mask_mean(fT, land_mask)
    record["fPAR"] = _spatial_mask_mean(fPAR, land_mask)
    record["lai"] = _spatial_mask_mean(lai, land_mask)
    record["SWdn"] = _spatial_mask_mean(SWdn, land_mask)
    record["ts_C"] = _spatial_mask_mean(np.asarray(ts, dtype=float) - TMELT, land_mask)
    if mrso is not None:
        record["mrso"] = _spatial_mask_mean(mrso, land_mask)
    if evap is not None:
        record["evap"] = _spatial_mask_mean(evap, land_mask)

    rec_mean = _spatial_mask_mean(rec_gppl, land_mask)
    record["gppl_reconstructed"] = rec_mean
    if rec_mean is not None and record["gppl"] not in (None, 0.0):
        record["gppl_recon_relerr"] = abs(rec_mean - record["gppl"]) / abs(record["gppl"])

    if avg_gpp is None:
        record["avg_gpp"] = record["gpp"]
    return record


def attribution_vs_reference(record, reference):
    """Fractional (log) contributions of each light factor vs a reference run."""
    if record.get("crashed") or reference is None or reference.get("crashed"):
        return {
            "dln": {name: None for name in LIGHT_FACTORS + ("gppl", "gppw", "gpp")},
            "dominant_light_term": None,
        }
    dln_factors = {
        name: dln(record.get(name), reference.get(name)) for name in LIGHT_FACTORS
    }
    dln_all = dict(dln_factors)
    for name in ("gppl", "gppw", "gpp"):
        dln_all[name] = dln(record.get(name), reference.get(name))
    return {
        "dln": dln_all,
        "dominant_light_term": dominant_term(dln_factors),
    }

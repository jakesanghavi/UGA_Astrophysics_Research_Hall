"""Atmosphere composition and gas-constant knobs for the production pipeline.

Composition (`atmos_type`) and thermodynamics (`physics_mode`) are separate:

- Composition picks the partial-pressure mix (and whether H/He evolution
  overwrites pH2/pHe). Venus vs Mars are both CO2-dominated but different
  mixes; pass ``atmos_params`` to tweak either without a new helper file.
- Physics mode decides how PlaSim's ``gascon`` (R) and ``akap`` (κ = R/Cp)
  are set. ExoPlaSim derives ``gascon`` from partial pressures but leaves
  ``akap`` hardcoded at 0.286 (Earth diatomic) unless we write it ourselves.

Diagnostic default knobs stay ``evolved`` / ``n2`` / ``co2`` so the GPP
diagnostic grid is unchanged. ``earth`` aliases ``n2``; ``venus`` aliases
``co2``. ``mars`` is the Curiosity/Mahaffy dry-air mix at ~1 bar (same
normalization as the Venus-like preset, which is also ~1 bar of Venus air
rather than 92 bar).
"""

GAS_KEYS = (
    "pH2", "pHe", "pN2", "pO2", "pCO2", "pAr", "pNe", "pKr", "pH2O", "pCH4",
)

# Same universal gas constant ExoPlaSim uses: gascon [J K^-1 kg^-1] = R_UNIV / mmw.
R_UNIV = 8314.46261815324
R_MOLAR = 8.31446261815324  # J mol^-1 K^-1

# Molecular weights [g mol^-1], matching exoplasim.constants.smws (plus CH4).
SMWS = {
    "H2": 2.01588,
    "He": 4.002602,
    "N2": 28.0134,
    "O2": 31.9988,
    "CO2": 44.01,
    "Ar": 39.948,
    "Ne": 20.1797,
    "Kr": 83.798,
    "H2O": 18.01528,
    "CH4": 16.04246,
}

# Molar heat capacity at constant pressure [J mol^-1 K^-1].
# Diatomic = 7/2 R, monatomic = 5/2 R. CO2 is set so pure CO2 reproduces
# PlaSim's Mars kappa (akap = 0.2273 in p_mars.f90).
CP_MOLAR = {
    "H2": 3.5 * R_MOLAR,
    "He": 2.5 * R_MOLAR,
    "N2": 3.5 * R_MOLAR,
    "O2": 3.5 * R_MOLAR,
    "CO2": R_MOLAR / 0.2273,
    "Ar": 2.5 * R_MOLAR,
    "Ne": 2.5 * R_MOLAR,
    "Kr": 2.5 * R_MOLAR,
    "H2O": 33.6,
    "CH4": 35.7,
}

# Earth-like mix (former model_helpers.py / model_helpers_n2.py).
EARTHLIKE_GASES = {
    "pH2": 0.0,
    "pHe": 5.24e-6,
    "pN2": 0.78084,
    "pO2": 0.20946,
    "pCO2": 330.0e-6,
    "pAr": 9.34e-3,
    "pNe": 18.18e-6,
    "pKr": 1.14e-6,
    "pH2O": 0.01,
    "pCH4": 0.0,
}

# Venus-like mix (former model_helpers_co2.py): Venus mole fractions at ~1 bar.
VENUSLIKE_GASES = {
    "pH2": 0.0,
    "pHe": 0.0,
    "pN2": 0.0341,
    "pO2": 69.3e-6,
    "pCO2": 0.964,
    "pAr": 18.6e-6,
    "pNe": 4.31e-6,
    "pKr": 0.0,
    "pH2O": 0.0,
    "pCH4": 0.0,
}

# Mars dry-air mix (Curiosity / Mahaffy et al. 2013), normalized to ~1 bar
# so Venus vs Mars is a composition tweak, not a 6 mbar vs 1 bar pressure leap.
# Override via atmos_params (e.g. scale all gases by 6.36e-3) for true Mars p_surf.
MARSLIKE_GASES = {
    "pH2": 0.0,
    "pHe": 0.0,
    "pN2": 0.0259,
    "pO2": 0.00161,
    "pCO2": 0.951,
    "pAr": 0.0194,
    "pNe": 0.0,
    "pKr": 0.0,
    "pH2O": 0.0,
    "pCH4": 0.0,
}

PRESETS = {
    "evolved": {
        "gases": EARTHLIKE_GASES,
        "apply_hhe": True,
        "label": "H/He evolution",
    },
    "n2": {
        "gases": EARTHLIKE_GASES,
        "apply_hhe": False,
        "label": "Earthlike N2/O2",
    },
    "co2": {
        "gases": VENUSLIKE_GASES,
        "apply_hhe": False,
        "label": "Venuslike CO2",
    },
    "mars": {
        "gases": MARSLIKE_GASES,
        "apply_hhe": False,
        "label": "Marslike CO2",
    },
}

# GPP-diagnostic default grid: the original three knobs, not mars.
ATMOS_TYPES = ("evolved", "n2", "co2")
PRESET_ALIASES = {"earth": "n2", "venus": "co2"}
PHYSICS_MODES = ("earth", "mars", "other")


def resolve_atmos_type(atmos_type):
    """Map aliases (earth, venus) onto canonical PRESETS keys."""
    key = PRESET_ALIASES.get(atmos_type, atmos_type)
    if key not in PRESETS:
        known = list(ATMOS_TYPES) + ["mars"] + list(PRESET_ALIASES)
        raise ValueError(f"Unknown atmos_type {atmos_type!r}; expected one of {known}")
    return key


def resolve_physics_mode(physics_mode):
    if physics_mode not in PHYSICS_MODES:
        raise ValueError(
            f"Unknown physics_mode {physics_mode!r}; expected one of {PHYSICS_MODES}"
        )
    return physics_mode


def apply_atmos_preset(params, atmos_type, overrides=None):
    """Overwrite gas partial pressures on a planet-params dict. Returns the preset.

    ``overrides`` is an optional dict of pN2/pCO2/... values applied on top of
    the preset, so Venus vs Mars (both CO2-dominated) can be tweaked from
    run_model.py without a new helper file.
    """
    key = resolve_atmos_type(atmos_type)
    preset = PRESETS[key]
    params.update(preset["gases"])
    if overrides:
        for gas_key, value in overrides.items():
            if value is None:
                continue
            params[gas_key] = value
    return preset


def total_pressure_bar(params):
    """Surface pressure [bar] as the sum of configured partial pressures."""
    return float(sum(float(params.get(k, 0.0) or 0.0) for k in GAS_KEYS))


def co2_ppmv_from_params(params):
    """CO2 mixing ratio in ppmv, matching ExoPlaSim's pCO2 / P_total conversion."""
    p_co2 = float(params.get("pCO2", 0.0) or 0.0)
    p_tot = total_pressure_bar(params)
    if p_tot <= 0.0:
        return 0.0
    return (p_co2 / p_tot) * 1.0e6


def mole_fractions(params):
    """Dry-air mole fractions from positive partial pressures."""
    amounts = {}
    for key in GAS_KEYS:
        gas = key[1:]
        pressure = float(params.get(key, 0.0) or 0.0)
        if pressure > 0.0 and gas in SMWS:
            amounts[gas] = pressure
    total = sum(amounts.values())
    if total <= 0.0:
        raise ValueError("no gases with positive partial pressure")
    return {gas: pressure / total for gas, pressure in amounts.items()}


def mmw_from_params(params):
    """Mean molecular weight [g mol^-1] from partial pressures."""
    fractions = mole_fractions(params)
    return sum(fractions[gas] * SMWS[gas] for gas in fractions)


def thermo_from_params(params):
    """gascon = R_UNIV/mmw and akap = R/Cp for the current mix.

    ``gascon`` matches ExoPlaSim's configure() formula. ``akap`` is what
    ExoPlaSim does *not* derive (p_exo.f90 hardcodes 0.286).
    """
    fractions = mole_fractions(params)
    mmw = sum(fractions[gas] * SMWS[gas] for gas in fractions)
    cp_molar = sum(fractions[gas] * CP_MOLAR[gas] for gas in fractions)
    return {
        "mmw": mmw,
        "gascon": R_UNIV / mmw,
        "akap": R_MOLAR / cp_molar,
    }


def apply_physics_mode(local_params, physics_mode):
    """Set gascon/akap on ``local_params`` and return exo.Model kwargs.

    - ``earth`` (default): do not pass gascon/akap. ExoPlaSim derives gascon
      from partial pressures; akap stays the compiled Earth/exo 0.286.
    - ``mars``: ``Model(mars=True)`` so PlaSim compiles p_mars.f90. That module
      also changes calendar, ozone, soil moisture, and orbit defaults — use
      only for actual Mars, not for a Mars-composition planet at 1 AU.
    - ``other``: write gascon = 8314.46/mmw and akap = R/Cp from the mix
      (needed for Venus/Mars-composition planets that should not use p_mars).
    """
    mode = resolve_physics_mode(physics_mode)
    local_params.pop("gascon", None)
    otherargs = dict(local_params.get("otherargs") or {})
    otherargs.pop("AKAP@planet_namelist", None)

    if mode == "earth":
        if otherargs:
            local_params["otherargs"] = otherargs
        else:
            local_params.pop("otherargs", None)
        return {"mars": False}

    if mode == "mars":
        if otherargs:
            local_params["otherargs"] = otherargs
        else:
            local_params.pop("otherargs", None)
        return {"mars": True}

    thermo = thermo_from_params(local_params)
    local_params["gascon"] = thermo["gascon"]
    otherargs["AKAP@planet_namelist"] = str(thermo["akap"])
    local_params["otherargs"] = otherargs
    return {"mars": False}


def regime_file_tag(run_mode, atmos_type, physics_mode):
    """Filename tag so composition/physics regimes do not clobber each other.

    The production default (n2 + earth physics) keeps the historical names
    ``_massonly2`` / ``_normal_n2`` so existing output files still match.
    """
    atmos = resolve_atmos_type(atmos_type)
    physics = resolve_physics_mode(physics_mode)
    mode_tag = "_massonly" if run_mode == "mass_only" else "_normal"
    if atmos == "n2" and physics == "earth":
        return "_massonly2" if run_mode == "mass_only" else "_normal_n2"
    extra = f"_{atmos}" if physics == "earth" else f"_{atmos}_{physics}"
    return mode_tag + extra


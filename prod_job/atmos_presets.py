"""Atmosphere-composition knobs used by the GPP-term diagnostic.

These match the three `model_helpers*` variants on main:

- ``evolved`` — `model_helpers.py` (no suffix): Earth-like mix, then overwrite
  pH2/pHe with the H/He envelope left by `evolve_atmosphere`.
- ``n2`` — `model_helpers_n2.py`: Earth-like N2/O2 mix, H/He evolution off.
- ``co2`` — `model_helpers_co2.py`: Venus-like CO2 mix, H/He evolution off.

Partial pressures are in bar.
"""

GAS_KEYS = (
    "pH2", "pHe", "pN2", "pO2", "pCO2", "pAr", "pNe", "pKr", "pH2O", "pCH4",
)

# Earth-like mix from model_helpers.py / model_helpers_n2.py.
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

# Venus-like mix from model_helpers_co2.py.
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
}

ATMOS_TYPES = tuple(PRESETS.keys())


def apply_atmos_preset(params, atmos_type):
    """Overwrite gas partial pressures on a planet-params dict. Returns the preset."""
    if atmos_type not in PRESETS:
        raise ValueError(
            f"Unknown atmos_type {atmos_type!r}; expected one of {ATMOS_TYPES}"
        )
    preset = PRESETS[atmos_type]
    params.update(preset["gases"])
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

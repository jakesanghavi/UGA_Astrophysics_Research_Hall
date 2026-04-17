import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import json
import numpy as np

rows = []
mp_values = ["01", "025", "05", "075", "1", "15", "2", "3", "4"]  # corresponds to 0.5, 1.0, 1.5
mp_numeric = [float(m) if (not m.startswith("0") and m != "15") else float(m[0] + "." + m[1:]) for m in mp_values]
RESOLUTION = "T21"
to_append = RESOLUTION if RESOLUTION == "T42" else ""

data_files = {mp: f"16cpus_test_{mp}{to_append}.json" for mp in mp_values}

# choose which index from the JSON list to plot (0 or 1)
VALUE_INDEX = 0

# ---- LOAD DATA ----
all_data = {}

for mp, filename in data_files.items():
    with open(filename, "r") as f:
        all_data[mp] = json.load(f)

mstar_keys = sorted(all_data[mp_values[0]].keys(), key=float)

# assume same AU keys structure across files
au_keys_per_mstar = {
    m: sorted(all_data[mp_values[0]][m].keys(), key=float)
    for m in mstar_keys
}

for mp, filename in data_files.items():
    with open(filename, "r") as f:
        all_data[mp] = json.load(f)

for mp_str, mp_val in zip(mp_values, mp_numeric):
    for mstar in mstar_keys:
        for au in au_keys_per_mstar[mstar]:
            
            val = all_data[mp_str][mstar][au][VALUE_INDEX]
            val /= 5.637932898316933e-10  # same normalization
            
            rows.append({
                "GPP": val,
                "PlanetMass": mp_val,
                "StellarMass": float(mstar),
                "Distance": float(au)
            })

df = pd.DataFrame(rows)

model = ols(
    'GPP ~ C(PlanetMass) + C(StellarMass) + C(Distance)',
    data=df
).fit()

anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

model = ols(
    'GPP ~ PlanetMass * StellarMass * Distance',
    data=df
).fit()

print(model.summary())
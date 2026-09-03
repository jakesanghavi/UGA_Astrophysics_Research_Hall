import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("gpp_statistical_test_df.csv")

NORMALIZATION = 5.637932898316933e-10

df = df.rename(columns={
    "GPP": "gpp",
    "PlanetMass": "mp",
    "StellarMass": "mstar",
    "Distance": "au"
})

# ensure numeric types
df["mp"] = df["mp"].astype(float)
df["mstar"] = df["mstar"].astype(float)
df["au"] = df["au"].astype(float)

mp_values = sorted(df["mp"].unique())
mp_numeric = mp_values

mstar_keys = sorted(df["mstar"].unique())
au_keys_per_mstar = {
    m: sorted(df[df["mstar"] == m]["au"].unique())
    for m in mstar_keys
}

fig, axes = plt.subplots(len(mstar_keys), 3,
                         figsize=(15, 10), sharex=True)

for i, mstar in enumerate(mstar_keys):
    au_keys = au_keys_per_mstar[mstar]

    for j, au in enumerate(au_keys):
        ax = axes[i, j]

        subset = df[(df["mstar"] == mstar) & (df["au"] == au)]

        values = []
        for mp in mp_values:
            row = subset[subset["mp"] == mp]

            if not row.empty:
                val = row["gpp"].iloc[0] / NORMALIZATION
            else:
                val = np.nan  # or 0

            values.append(val)

        x = np.arange(len(mp_values))
        ax.bar(x, values)

        ax.set_title(f"M*={mstar}, AU={np.round(au, 3)}")

        if i == len(mstar_keys) - 1:
            ax.set_xticks(x)
            ax.set_xticklabels(mp_numeric)

        if j == 0:
            ax.set_ylabel("Normalized GPP")

fig.suptitle(
    "Gross Primary Production by M*, AU, and Mp.\n"
    "Bars normalized to Earth-like case (M*=1, AU=1.13, Mp=1.0)."
)

plt.tight_layout()
plt.show()
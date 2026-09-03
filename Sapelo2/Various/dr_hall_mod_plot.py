import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# df = pd.read_csv("gpp_statistical_test_df.csv")
df = pd.read_csv("statistical_data.csv")

# NORMALIZATION = 5.637932898316933e-10
NORMALIZATION = 1

df = df.rename(columns={
    "GPP": "gpp",
    "PlanetMass": "mp",
    "StellarMass": "mstar",
    "Distance": "au",
    "Flux": "flux"
})

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

group1 = [0.1, 0.5, 1.0]
group2 = [1.25, 1.5, 2.0]
near1_group = [0.7, 0.8, 0.9, 1.0, 1.1]

def plot_grid(mstar_subset, yscale="linear", sharey=False, title_suffix=""):
    fig, axes = plt.subplots(len(mstar_subset), 3,
                             figsize=(15, 10),
                             sharex=True,
                             sharey=sharey)

    all_plot_values = []
    global_min, global_max = np.inf, -np.inf

    for i, mstar in enumerate(mstar_subset):
        au_keys = au_keys_per_mstar.get(mstar, [])

        for j, au in enumerate(au_keys):
            ax = axes[i, j]
            ax.tick_params(axis='x', which='major', labelsize=8)

            subset = df[(df["mstar"] == mstar) & (df["au"] == au)]

            values = []
            for mp in mp_values:
                row = subset[subset["mp"] == mp]

                if not row.empty:
                    val = row["gpp"].iloc[0] / NORMALIZATION
                else:
                    val = np.nan

                values.append(val)

            values = np.array(values)

            # Prevent log of 0
            values = np.clip(values, 1e-20, None)

            all_plot_values.extend(values)
            global_min = min(global_min, np.nanmin(values))
            global_max = max(global_max, np.nanmax(values))

            x = np.arange(len(mp_values))
            ax.bar(x, values)

            ax.set_title(f"M*={mstar}, AU={np.round(au, 2)}")

            if i == len(mstar_subset) - 1:
                ax.set_xticks(x)
                ax.set_xticklabels(mp_numeric)

            if j == 0:
                ax.set_ylabel("Normalized GPP")

            if yscale == "log":
                ax.set_yscale("log")

    threshold = 1e-15
    all_plot_values = np.array(all_plot_values)
    valid_mask = all_plot_values > threshold

    if sharey:
        if np.any(valid_mask):
            adjusted_min = all_plot_values[valid_mask].min()
        else:
            adjusted_min = threshold

        adjusted_max = global_max if global_max > threshold else threshold

        for ax_row in axes:
            for ax in ax_row:
                ax.set_ylim(adjusted_min, adjusted_max)

        fig.suptitle(
            f"Gross Primary Production by M*, AU, and Mp.\n{title_suffix}"
        )

    plt.tight_layout()

    filename = (
        title_suffix.replace(' ', '_')
        .replace('(', '')
        .replace(')', '')
        .replace(',', '')
        + ".png"
    )

    plt.savefig(filename, dpi=150)
    plt.close(fig)

plot_grid(group1, yscale="linear", sharey=True,
          title_suffix="Group 1 (0.1–1.0 M☉), shared linear scale")

plot_grid(group2, yscale="linear", sharey=True,
          title_suffix="Group 2 (1.25–2.0 M☉), shared linear scale")

plot_grid(group1, yscale="log", sharey=True,
          title_suffix="Group 1, log10 scale (shared)")

plot_grid(group2, yscale="log", sharey=True,
          title_suffix="Group 2, log10 scale (shared)")

plot_grid(group1, yscale="linear", sharey=False,
          title_suffix="Group 1, independent linear scales")

plot_grid(group2, yscale="linear", sharey=False,
          title_suffix="Group 2, independent linear scales")

plot_grid(near1_group, yscale="linear", sharey=True,
          title_suffix="Near 1 AU (0.7–1.1 M☉), shared linear scale")

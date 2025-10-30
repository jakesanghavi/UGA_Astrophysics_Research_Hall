import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt

columns = ["mass", "avg_veg", 'total_veg']
df = pd.DataFrame(columns=columns + ['f_init'])

for f_init in [0.05, 0.1, 0.15]:
    data = None
    with open(f"veg_json_init_f_{f_init}.json") as f:
        data = json.load(f)
    df_mini = pd.DataFrame.from_dict(data, orient="index").reset_index()
    df_mini.columns = columns
    df_mini['f_init'] = f_init
    
    with open(f"veg_json_init_f_{f_init}_step_0.1.json") as f:
        data = json.load(f)
    df_mini2 = pd.DataFrame.from_dict(data, orient="index").reset_index()
    df_mini2.columns = columns
    df_mini2['f_init'] = f_init
    
    df = pd.concat([df, df_mini, df_mini2])
    
df = df.reset_index(drop=True)
df['mass'] = df['mass'].astype(float)
df = df.sort_values(by=['f_init', 'mass'])
    
sns.set(style="whitegrid")

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10), sharex=True)

# --- 1️⃣ Top plot: avg_veg ---
sns.barplot(
    data=df,
    x="mass",
    y="avg_veg",
    hue="f_init",
    ax=axes[0],
    palette="Blues_d",
    dodge=True
)
axes[0].set_title("Average Vegetation by Mass and f_init")
axes[0].set_ylabel(r"Average surface land vegetation $kg C m^{-2}$")
axes[0].legend(title="f_init", loc="upper left")

# --- 2️⃣ Bottom plot: total_veg ---
sns.barplot(
    data=df,
    x="mass",
    y="total_veg",
    hue="f_init",
    ax=axes[1],
    palette="Reds_d",
    dodge=True
)
axes[1].set_title("Total Vegetation by Mass and f_init")
axes[1].set_ylabel(r"Total surface land $kg C$")
axes[1].set_xlabel("mass")
axes[1].legend(title="f_init", loc="upper left")

plt.tight_layout()
plt.show()

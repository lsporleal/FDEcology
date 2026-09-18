
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 13:17:49 2026

@author: leasp
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# LOAD DATA
# ============================================================

df_main = pd.read_csv(
    "C:/Users/37768378/Documents/Ecology/Graphic Process Detrended Version/SIF_pentad_5day_with_SIF_anomaly.csv"
)

df_geo = pd.read_csv(
    "C:/Users/37768378/Documents/Ecology/EcoRegions/Coord_LCCS_EcoRegions.csv"
)

# Keep only one land cover/ecoregion per grid
df_geo = (
    df_geo[
        ["latitude", "longitude", "lccs_class", "NA_L1NAME"]
    ]
    .drop_duplicates(subset=["latitude", "longitude"])
)

# Merge
df = pd.merge(
    df_main,
    df_geo,
    on=["latitude", "longitude"],
    how="inner"
)

# ============================================================
# CLEAN DATA
# ============================================================

df = df.dropna(
    subset=[
        "SIF_anomaly_detrended",
        "lccs_class",
        "NA_L1NAME",
        "phase_name"
    ]
)

df = df[df["NA_L1NAME"] != "WATER"]
df["phase_name"] = df["phase_name"].str.lower()

# Drop unwanted LCCS classes
df = df[
    ~df["lccs_class"].isin([
        "Tree Cover - Flooded - Fresh Water",
        "Tree Cover - Flooded – Salt Water",
        "Water Body",
        "Permanent Snow/Ice",
        "Shrub/Herbaceous - Flooded"
    ])
]

# Check lccs classes
values_list = df["lccs_class"].unique().tolist()
print(values_list)

# Consistent category order
df["lccs_class"] = df["lccs_class"].astype("category")
df["NA_L1NAME"] = df["NA_L1NAME"].astype("category")
# ============================================================
# COLOR SCALE
# ============================================================

vmin = -1
vmax = 1

# ============================================================
# ALL PHASES IN ONE FIGURE
# ============================================================

phase_order = ["normal", "onset", "persistence", "cooldown"]
phases = [p for p in phase_order if p in df["phase_name"].unique()]

all_ecoregions = sorted(df["NA_L1NAME"].unique())
all_lccs = sorted(df["lccs_class"].unique())

# Grey for missing combinations
cmap = plt.cm.RdBu_r.copy()
cmap.set_bad(color="lightgrey")

fig, axes = plt.subplots(
    2,
    2,
    figsize=(20, 14),
    dpi=300
)

axes = axes.flatten()

for i, phase in enumerate(phases):

    ax = axes[i]

    subset = df[df["phase_name"] == phase]

    heatmap_data = (
        subset.pivot_table(
            index="NA_L1NAME",
            columns="lccs_class",
            values="SIF_anomaly_detrended",
            aggfunc="mean"
        )
        .reindex(
            index=all_ecoregions,
            columns=all_lccs
        )
    )

    im = ax.imshow(
        np.ma.masked_invalid(heatmap_data.values),
        cmap=cmap,
        aspect="auto",
        vmin=vmin,
        vmax=vmax
    )

    ax.set_title(
        phase.title(),
        fontsize=22,
        fontweight="bold"
    )

    # Left column gets y labels
    if i in [0, 2]:
        ax.set_yticks(np.arange(len(all_ecoregions)))
        ax.set_yticklabels(
            all_ecoregions,
            fontsize=18
        )
    else:
        ax.set_yticks([])

    # Bottom row gets x labels
    if i in [2, 3]:
        ax.set_xticks(np.arange(len(all_lccs)))
        ax.set_xticklabels(
            all_lccs,
            rotation=90,
            fontsize=18
        )
    else:
        ax.set_xticks([])

# ============================================================
# Shared labels
# ============================================================

fig.supxlabel(
    "Land Cover",
    fontsize=24,
    fontweight="bold",
    x=0.55,
    y=-0.02
)

fig.supylabel(
    "Ecoregion",
    fontsize=24,
    fontweight="bold",
    x=-0.06,
    y=0.54
)

# ============================================================
# Layout
# ============================================================

plt.subplots_adjust(
    left=0.18,
    right=0.88,
    bottom=0.28,
    top=0.92,
    wspace=0.12,
    hspace=0.18
)

# ============================================================
# Colorbar
# ============================================================

cbar_ax = fig.add_axes([0.90, 0.15, 0.025, 0.70])

cbar = fig.colorbar(
    im,
    cax=cbar_ax
)

cbar.set_label(
    "Mean SIF Anomaly",
    fontsize=22,
    fontweight="bold"
)

cbar.ax.tick_params(
    labelsize=18
)

# ============================================================
# Title
# ============================================================

fig.suptitle(
    "Flash Drought Impact by Phase (SIF)",
    fontsize=28,
    fontweight="bold"
)

plt.show()

# ============================================================
# SAVE TABLES TO EXCEL (SIF)
# ============================================================

with pd.ExcelWriter("Mean_SIF_Anomaly_By_Phase.xlsx") as writer:

    for phase in phases:

        subset = df[df["phase_name"] == phase]

        table = (
            subset.pivot_table(
                index="NA_L1NAME",
                columns="lccs_class",
                values="SIF_anomaly_detrended",
                aggfunc="mean"
            )
            .reindex(
                index=all_ecoregions,
                columns=all_lccs
            )
            .round(3)
        )

        table.to_excel(writer, sheet_name=phase[:31])

print("Saved: Mean_SIF_Anomaly_By_Phase.xlsx")

# ============================================================
# SAVE SUMMARY BY LC TO CSV
# ============================================================

summary = (
    df.groupby(
        ["lccs_class", "phase_name"],
        observed=True
    )["SIF_anomaly_detrended"]
    .agg(
        count="count",
        mean="mean"
    )
    .reset_index()
)

summary.to_csv(
    "SIF_Anomaly_Detrended_By_LCCS_Phase.csv",
    index=False
)

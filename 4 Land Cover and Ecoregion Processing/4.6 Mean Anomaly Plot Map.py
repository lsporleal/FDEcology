# -*- coding: utf-8 -*-
"""
Created on Wed May 13 09:58:20 2026

@author: azs03
"""

# -*- coding: utf-8 -*-
"""
Plot spatial maps of mean SIF anomaly by flash drought phase.
Input:
    results/SIF_Anomaly_Stats/mean_anomaly.csv
Output:
    results/SIF_Anomaly_Stats/plots/mean_SIF_anomaly_by_phase_US.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature


# =========================================================
# Config
# =========================================================
INPUT_CSV = "results/SIF_Anomaly_Stats/mean_anomaly.csv"
OUT_DIR = "results/SIF_Anomaly_Stats/plots"
os.makedirs(OUT_DIR, exist_ok=True)

# Set YEAR = None to average across all years
# Set YEAR = 2012, for example, to plot only one year
YEAR = None

PHASE_COLS = {
    "Normal": "Mean_Normal",
    "Onset": "Mean_Onset",
    "Persistence": "Mean_Persistence",
    "CoolDown": "Mean_CoolDown",
}

US_EXTENT = [-125, -66.5, 24, 50]  # CONUS


# =========================================================
# Load and prepare data
# =========================================================
INPUT_CSV = r"C:/Users/leasp/Documents/UNL/Group Work - FD Ecology/NDVI_anomaly_all_grids.csv"

df = pd.read_csv(INPUT_CSV)

df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["NDVI_anomaly"] = pd.to_numeric(df["NDVI_anomaly"], errors="coerce")

df = df.dropna(subset=["latitude", "longitude", "NDVI_anomaly"])

# Keep CONUS only
df = df[
    (df["longitude"] >= US_EXTENT[0]) &
    (df["longitude"] <= US_EXTENT[1]) &
    (df["latitude"] >= US_EXTENT[2]) &
    (df["latitude"] <= US_EXTENT[3])
].copy()

# Standardize phase names
df["phase_name"] = df["phase_name"].replace({
    "normal": "Normal",
    "onset": "Onset",
    "persistence": "Persistence",
    "cooldown": "CoolDown",
    "Cooldown": "CoolDown"
})

# Mean NDVI anomaly by grid cell and phase
df_all_years = (
    df.groupby(
        ["latitude", "longitude", "phase_name"],
        as_index=False
    )["NDVI_anomaly"]
    .mean()
    .pivot_table(
        index=["latitude", "longitude"],
        columns="phase_name",
        values="NDVI_anomaly"
    )
    .reset_index()
)

df_all_years.columns.name = None

df_all_years = df_all_years.rename(columns={
    "latitude": "Latitude",
    "longitude": "Longitude",
    "Normal": "Mean_Normal",
    "Onset": "Mean_Onset",
    "Persistence": "Mean_Persistence",
    "CoolDown": "Mean_CoolDown"
})

# =========================================================
# Shared color scale
# =========================================================
all_vals = pd.concat(
    [df_all_years[col] for col in PHASE_COLS.values()],
    ignore_index=True
)

vmax = np.nanpercentile(np.abs(all_vals), 98)
vmin = -vmax

print("Color scale:", vmin, vmax)
print("Points plotted:", df_all_years[["Latitude", "Longitude"]].drop_duplicates().shape[0])


# =========================================================
# Function to plot one figure
# =========================================================
def plot_phase_map(df_plot, title_suffix, out_name):
    all_vals = pd.concat(
        [df_plot[col] for col in PHASE_COLS.values()],
        ignore_index=True
    )

    vmax = np.nanpercentile(np.abs(all_vals), 98)
    vmin = -vmax

    fig, axes = plt.subplots(
        2, 2,
        figsize=(14, 8),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )

    axes = axes.ravel()
    last_sc = None
    
    for i, (ax, (phase, col)) in enumerate(zip(axes, PHASE_COLS.items())):
        ax.set_extent(US_EXTENT, crs=ccrs.PlateCarree())
    
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.STATES, linewidth=0.3)
    
        sub = df_plot.dropna(subset=[col])
    
        last_sc = ax.scatter(
            sub["Longitude"],
            sub["Latitude"],
            c=sub[col],
            s=8,
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree()
        )
    
        ax.set_title(phase, fontsize=12)
    
        gl = ax.gridlines(draw_labels=True, linewidth=0.2, alpha=0.4)
        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = i in [0, 2]      # only left column
        gl.bottom_labels = i in [2, 3]    # only bottom row
        
    plt.subplots_adjust(wspace=0.05, hspace=0.2)

    cbar = fig.colorbar(
        last_sc,
        ax=axes,
        orientation="horizontal",
        fraction=0.05,
        pad=0.08
    )
    cbar.set_label("Mean NDVI anomaly")

    fig.suptitle(
    f"Spatial Pattern of Mean NDVI Anomaly by Flash Drought Phase\n{title_suffix}",
    fontsize=15
)

    out_fp = os.path.join(OUT_DIR, out_name)
    plt.savefig(out_fp, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", out_fp)


# =========================================================
# Plot mean across all years
# =========================================================
plot_phase_map(
    df_plot=df_all_years,
    title_suffix="Mean across all years",
    out_name="mean_NDVI_anomaly_by_phase_all_years_US.png"
)


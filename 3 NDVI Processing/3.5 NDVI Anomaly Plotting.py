# -*- coding: utf-8 -*-
"""
Plot spatial maps of mean NDVI anomaly by flash drought phase.

Produces, for EACH anomaly type (detrended, nondetrended):
  1) one figure: mean across all years
  2) one figure per year (auto-detected from the data)

Plot format (figsize, colors, fonts, gridlines, colorbar, spacing) is
identical to the SIF version -- only the SIF -> NDVI labels/paths changed.

Input:
    NDVI_Anomaly_Stats/<ANOMALY_TYPE>/mean_anomaly.csv
Output:
    NDVI_Anomaly_Stats/<ANOMALY_TYPE>/plots/mean_NDVI_anomaly_by_phase_all_years_US.png
    NDVI_Anomaly_Stats/<ANOMALY_TYPE>/plots/mean_NDVI_anomaly_by_phase_<year>_US.png
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
BASE_DIR = r"D:\UNL\FD&Ecosystem\DATA\NEW1\NDVI_Anomaly_Stats"

ANOMALY_TYPES = ["detrended", "nondetrended"]

PHASE_COLS = {
    "Normal": "normal",
    "Onset": "onset",
    "Persistence": "persistence",
    "CoolDown": "cooldown",
}

US_EXTENT = [-125, -66.5, 24, 50]  # CONUS


# =========================================================
# Function to plot one figure (format unchanged from original)
# =========================================================
def plot_phase_map(df_plot, title_suffix, out_name, out_dir):
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

    out_fp = os.path.join(out_dir, out_name)
    plt.savefig(out_fp, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", out_fp)


# =========================================================
# Process one anomaly type (detrended or nondetrended)
# =========================================================
def process_anomaly_type(anomaly_type):
    input_csv = os.path.join(BASE_DIR, anomaly_type, "mean_anomaly.csv")
    out_dir = os.path.join(BASE_DIR, anomaly_type, "plots")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== Processing anomaly type: {anomaly_type} ===")

    # -----------------------------------------------------
    # Load and prepare data
    # -----------------------------------------------------
    df = pd.read_csv(input_csv)

    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

    for col in PHASE_COLS.values():
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Latitude", "Longitude"])

    # Keep CONUS only
    df = df[
        (df["Longitude"] >= US_EXTENT[0]) &
        (df["Longitude"] <= US_EXTENT[1]) &
        (df["Latitude"] >= US_EXTENT[2]) &
        (df["Latitude"] <= US_EXTENT[3])
    ].copy()

    # Average yearly mean anomalies across all years at each point
    df_plot_all_years = (
        df.groupby(["Latitude", "Longitude"], as_index=False)
        [list(PHASE_COLS.values())]
        .mean()
    )

    # -----------------------------------------------------
    # Shared color scale (info only; recomputed inside plot_phase_map too)
    # -----------------------------------------------------
    all_vals = pd.concat([df_plot_all_years[col] for col in PHASE_COLS.values()], ignore_index=True)
    vmax = np.nanpercentile(np.abs(all_vals), 98)
    vmin = -vmax

    print("Color scale:", vmin, vmax)
    print("Points plotted:", df_plot_all_years[["Latitude", "Longitude"]].drop_duplicates().shape[0])

    # -----------------------------------------------------
    # 1) Plot mean across all years
    # -----------------------------------------------------
    plot_phase_map(
        df_plot=df_plot_all_years,
        title_suffix="Mean across all years",
        out_name="mean_NDVI_anomaly_by_phase_all_years_US.png",
        out_dir=out_dir
    )

    # -----------------------------------------------------
    # 2) Plot each year separately
    # -----------------------------------------------------
    years = sorted(df["Year"].dropna().astype(int).unique())

    for year in years:
        df_year = df[df["Year"] == year].copy()

        plot_phase_map(
            df_plot=df_year,
            title_suffix=str(year),
            out_name=f"mean_NDVI_anomaly_by_phase_{year}_US.png",
            out_dir=out_dir
        )


# =========================================================
# Main
# =========================================================
def main():
    for anomaly_type in ANOMALY_TYPES:
        process_anomaly_type(anomaly_type)


if __name__ == "__main__":
    main()

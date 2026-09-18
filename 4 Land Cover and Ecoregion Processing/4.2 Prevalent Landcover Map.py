# -*- coding: utf-8 -*-
"""
Created on Wed May 20 09:14:54 2026

@author: leasp
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------
csv_file = r"D:/UNL/DATA/CCI LandCoverGrid/FDCoord/Coord_LCCS_EcoRegions.csv"
OUT_DIR = r"D:\UNL\DATA\CCI LandCoverGrid\FDCoord"

# map extent: [west, east, south, north]
US_EXTENT = [-125, -66, 24, 50]

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = pd.read_csv(csv_file)

df = df.dropna(
    subset=["latitude", "longitude", "lccs_class"]
)

# --------------------------------------------------
# FIND MOST COMMON lccs_class
# ACROSS ALL YEARS
# --------------------------------------------------
dominant = (
    df.groupby(["latitude", "longitude"])["lccs_class"]
    .agg(lambda x: x.value_counts().idxmax())
    .reset_index()
)

# --------------------------------------------------
# CONVERT CLASSES TO NUMBERS
# --------------------------------------------------
classes = sorted(dominant["lccs_class"].unique())

class_to_num = {
    cls: i for i, cls in enumerate(classes)
}

dominant["class_num"] = dominant["lccs_class"].map(class_to_num)

# --------------------------------------------------
# PLOT
# --------------------------------------------------
fig, ax = plt.subplots(
    figsize=(14, 8),
    subplot_kw={"projection": ccrs.PlateCarree()}
)

# reduce whitespace manually
fig.subplots_adjust(
    top=0.98,
    right=0.80
)

ax.set_extent(
    US_EXTENT,
    crs=ccrs.PlateCarree()
)

# --------------------------------------------------
# BACKGROUND FEATURES
# --------------------------------------------------
ax.add_feature(cfeature.LAND, linewidth=0)
ax.add_feature(cfeature.OCEAN, linewidth=0)
ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.add_feature(cfeature.STATES, linewidth=0.3)

# --------------------------------------------------
# PLOT GRID CELLS
# --------------------------------------------------
sc = ax.scatter(
    dominant["longitude"],
    dominant["latitude"],
    c=dominant["class_num"],
    s=22,
    cmap="tab20",
    marker="s",
    linewidth=0,
    transform=ccrs.PlateCarree()
)

# --------------------------------------------------
# GRIDLINES
# --------------------------------------------------
gl = ax.gridlines(
    draw_labels=True,
    linewidth=0.2,
    alpha=0.4
)

gl.top_labels = False
gl.right_labels = False

# --------------------------------------------------
# LEGEND
# --------------------------------------------------
handles = [
    plt.Line2D(
        [0], [0],
        marker='s',
        color='none',
        markerfacecolor=plt.cm.tab20(
            i / max(len(classes)-1, 1)
        ),
        markersize=10,
        label=str(cls)
    )
    for i, cls in enumerate(classes)
]

legend = ax.legend(
    handles=handles,
    title="Dominant LCCS Class",
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    fontsize=9,
    title_fontsize=10,
    frameon=True,
    ncol=1
)

# --------------------------------------------------
# TITLE
# --------------------------------------------------
ax.set_title(
    "Dominant Land Cover Class Across All Years",
    fontsize=14,
    pad=4
)

# --------------------------------------------------
# SAVE
# --------------------------------------------------
out_fp = os.path.join(
    OUT_DIR,
    "dominant_lccs_all_years.png"
)

plt.savefig(
    out_fp,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:", out_fp)
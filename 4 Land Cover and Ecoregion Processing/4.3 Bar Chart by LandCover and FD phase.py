# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 19:18:31 2026

@author: leasp
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================================
# LOAD CSV
# ============================================================

df = pd.read_csv(
    "C:/Users/37768378/Documents/Ecology/Graphic Process Detrended Version/NDVI_Anomaly_Detrended_By_LCCS_Phase.csv"
)

# ============================================================
# FIX PHASE NAMES
# ============================================================

df["phase_name"] = df["phase_name"].replace({
    "onset": "Onset",
    "cooldown": "CoolDown",
    "normal": "Normal",
    "persistence": "Persistence"
})

# ============================================================
# PHASE ORDER AND COLORS
# ============================================================

phase_order = [
    "Normal",
    "Onset",
    "Persistence",
    "CoolDown"
]

phase_colors = {
    "Normal": "green",
    "Onset": "red",
    "Persistence": "yellow",
    "CoolDown": "blue"
}

# ============================================================
# PLOT STYLE
# ============================================================

sns.set_style("whitegrid")

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 24,
    "axes.labelsize": 20,
    "xtick.labelsize": 15,
    "ytick.labelsize": 16,
    "legend.fontsize": 15,
    "legend.title_fontsize": 17
})

# ============================================================
# CREATE FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(18, 11),
    dpi=300
)

# ============================================================
# BARPLOT
# ============================================================

sns.barplot(
    data=df,
    x="lccs_class",
    y="mean",
    hue="phase_name",
    hue_order=phase_order,
    palette=phase_colors,
    errorbar=None,
    edgecolor="black",
    linewidth=1.2,
    ax=ax
)

# ============================================================
# LABELS
# ============================================================

ax.set_xlabel(
    "Land Cover Class",
    fontsize=30,
    labelpad=12
)

ax.set_ylabel(
    "Mean Detrended NDVI Anomaly",
    fontsize=30,
    labelpad=12
)

ax.set_title(
    "Mean Detrended NDVI Anomaly by Land Cover Class and Flash Drought Phase",
    fontsize=30,
    pad=18
)

# Y-axis range and formatting
ax.set_ylim(-0.3, 0.3)

from matplotlib.ticker import FormatStrFormatter
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))


# ============================================================
# X-AXIS LABELS
# ============================================================

ax.tick_params(
    axis="x",
    labelrotation=90,
    labelsize=25
)

ax.tick_params(
    axis="y",
    labelsize=25
)

# ============================================================
# GRID
# ============================================================

ax.grid(
    True,
    axis="y",
    linestyle="--",
    alpha=0.5
)

ax.grid(
    False,
    axis="x"
)

# ============================================================
# LEGEND
# ============================================================

ax.legend(
    title="Phase",
    loc="lower right",
    fontsize=20,
    title_fontsize=25,
    frameon=True
)

# ============================================================
# LAYOUT
# ============================================================

fig.subplots_adjust(
    left=0.10,
    right=0.97,
    bottom=0.32,
    top=0.90
)

plt.show()
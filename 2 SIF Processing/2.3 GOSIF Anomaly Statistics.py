# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 14:08:50 2026

@author: 28355914

Compute yearly SIF anomaly statistics by phase -- for BOTH the detrended
and non-detrended anomaly, into separate subfolders.

Input:
    SIF_pentad_5day_with_SIF_anomaly.csv
    (must contain SIF_anomaly_detrended and SIF_anomaly_nondetrended)

Outputs in:
    SIF_Anomaly_Stats/detrended/
        mean_anomaly.csv
        median_anomaly.csv
        std_anomaly.csv
        min_anomaly.csv
        max_anomaly.csv
        count_anomaly.csv
    SIF_Anomaly_Stats/nondetrended/
        (same six files)
"""

import os
import pandas as pd
import numpy as np


# =========================================================
# Config
# =========================================================
INPUT_CSV = "results/SIF_pentad_5day_with_SIF_anomaly.csv"
OUT_DIR = "results/SIF_Anomaly_Stats"

PHASE_ORDER = ["Normal", "Onset", "Persistence", "CoolDown"]

# Which anomaly columns to process, and the subfolder each goes into
ANOMALY_COLUMNS = {
    "detrended": "SIF_anomaly_detrended",
    "nondetrended": "SIF_anomaly_nondetrended",
}

STATS = [
    ("Mean", "mean"),
    ("Median", "median"),
    ("Std", "std"),
    ("Min", "min"),
    ("Max", "max"),
    ("Count", "count"),
]


# =========================================================
# Helpers
# =========================================================
def make_phase_stat_table(df, value_col, stat_name, aggfunc):
    """
    Create wide table:
    Latitude, Longitude, Year,
    <Stat>_Normal, <Stat>_Onset, <Stat>_Persistence, <Stat>_CoolDown, <Stat>_all
    """
    # phase-wise stat
    phase_tbl = (
        df.groupby(["latitude", "longitude", "year", "phase_name"])[value_col]
        .agg(aggfunc)
        .unstack("phase_name")
        .reset_index()
    )

    # make sure all expected phase columns exist
    for ph in PHASE_ORDER:
        if ph not in phase_tbl.columns:
            phase_tbl[ph] = np.nan

    # reorder phase columns
    phase_tbl = phase_tbl[["latitude", "longitude", "year"] + PHASE_ORDER]

    # rename phase columns
    phase_tbl = phase_tbl.rename(columns={
        "latitude": "Latitude",
        "longitude": "Longitude",
        "year": "Year",
        "Normal": f"{stat_name}_Normal",
        "Onset": f"{stat_name}_Onset",
        "Persistence": f"{stat_name}_Persistence",
        "CoolDown": f"{stat_name}_CoolDown",
    })

    # overall yearly stat across all pentads
    all_tbl = (
        df.groupby(["latitude", "longitude", "year"])[value_col]
        .agg(aggfunc)
        .reset_index()
        .rename(columns={
            "latitude": "Latitude",
            "longitude": "Longitude",
            "year": "Year",
            value_col: f"{stat_name}_all",
        })
    )

    # merge
    out = phase_tbl.merge(
        all_tbl,
        on=["Latitude", "Longitude", "Year"],
        how="left"
    )

    return out


def compute_stats_for_column(raw, anomaly_col, sub_out_dir):
    """Clean + compute all six stat tables for one anomaly column, save to sub_out_dir."""
    os.makedirs(sub_out_dir, exist_ok=True)

    df = raw[["year", "latitude", "longitude", "phase_name", anomaly_col]].copy()
    df = df.rename(columns={anomaly_col: "SIF_anomaly"})

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["SIF_anomaly"] = pd.to_numeric(df["SIF_anomaly"], errors="coerce")
    df["phase_name"] = df["phase_name"].astype(str).str.strip()

    df = df.dropna(subset=["year", "latitude", "longitude", "SIF_anomaly"])
    df["year"] = df["year"].astype(int)

    for stat_name, aggfunc in STATS:
        stat_df = make_phase_stat_table(df, "SIF_anomaly", stat_name, aggfunc)
        fname = f"{stat_name.lower()}_anomaly.csv"
        stat_df.to_csv(os.path.join(sub_out_dir, fname), index=False)
        print(" -", os.path.join(sub_out_dir, fname))


# =========================================================
# Main
# =========================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    raw = pd.read_csv(INPUT_CSV)

    for label, anomaly_col in ANOMALY_COLUMNS.items():
        print(f"\n=== Computing stats for: {label} ({anomaly_col}) ===")
        sub_out_dir = os.path.join(OUT_DIR, label)
        compute_stats_for_column(raw, anomaly_col, sub_out_dir)

    print("\nSaved stats for both detrended and non-detrended anomalies in:", OUT_DIR)


if __name__ == "__main__":
    main()
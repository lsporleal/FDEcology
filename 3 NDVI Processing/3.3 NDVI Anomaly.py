# -*- coding: utf-8 -*-
"""
Computes BOTH:
  - NDVI_anomaly_detrended    : linear year-trend (per lat/lon/pentad, across
                                 all years) removed first, then residuals z-scored
  - NDVI_anomaly_nondetrended : raw NDVI z-scored directly per lat/lon/pentad,
                                 across all years -- no trend removal

VECTORIZED VERSION: no per-group Python loop / np.polyfit loop.
Regression slope/intercept computed via closed-form least-squares sums using
groupby().agg(), then merged back onto the full dataframe. This avoids the
runtime blowup seen with row-wise / per-group looping on ~15M-row datasets.
"""

import numpy as np
import pandas as pd
import os

# =========================================================
# Paths
# =========================================================
INPUT_FP  = r"D:\UNL\FD&Ecosystem\DATA\NEW1\NDVI_Time_Linear_Interpolated_with_phases.csv"
OUTPUT_DIR = r"D:\UNL\FD&Ecosystem\DATA\NEW1"
OUTPUT_FP  = os.path.join(OUTPUT_DIR, "NDVI_pentad_5day_with_NDVI_anomaly.csv")

GROUP_COLS = ["latitude", "longitude", "Pentad"]

# =========================================================
# Data loading
# =========================================================
def load_data(fp):
    print("Loading full dataset...")
    df = pd.read_csv(fp)
    df["NDVI"] = pd.to_numeric(df["NDVI"], errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    print(f"  Loaded {len(df):,} rows.")
    return df

# =========================================================
# Vectorized anomaly computation
# =========================================================
def compute_NDVI_anomaly(df):
    df = df.copy()
    valid_mask = df["NDVI"].notna()

    # ---------------------------------------------------
    # NON-DETRENDED: z-score of raw NDVI per (lat, lon, pentad)
    # ---------------------------------------------------
    print("Computing non-detrended anomaly (raw z-score)...")
    grp = df.groupby(GROUP_COLS, sort=False)["NDVI"]
    df["raw_count"] = grp.transform("count")          # ignores NaN
    df["raw_mean"]  = grp.transform("mean")            # ignores NaN
    df["raw_std"]   = grp.transform("std")              # ddof=1, ignores NaN

    ok_raw = valid_mask & (df["raw_count"] >= 2) & df["raw_std"].notna() & (df["raw_std"] != 0)
    df["NDVI_anomaly_nondetrended"] = np.nan
    df.loc[ok_raw, "NDVI_anomaly_nondetrended"] = (
        (df.loc[ok_raw, "NDVI"] - df.loc[ok_raw, "raw_mean"]) / df.loc[ok_raw, "raw_std"]
    )

    # ---------------------------------------------------
    # DETRENDED: remove linear Year-trend per (lat, lon, pentad)
    # using closed-form least squares (vectorized), then z-score residuals
    # ---------------------------------------------------
    print("Fitting per-group linear trend (vectorized least squares)...")
    valid = df.loc[valid_mask, GROUP_COLS + ["Year", "NDVI"]].copy()
    valid["xy"] = valid["Year"] * valid["NDVI"]
    valid["x2"] = valid["Year"] ** 2

    agg = valid.groupby(GROUP_COLS, sort=False).agg(
        n=("NDVI", "count"),
        sum_x=("Year", "sum"),
        sum_y=("NDVI", "sum"),
        sum_xy=("xy", "sum"),
        sum_x2=("x2", "sum"),
    ).reset_index()

    denom = (agg["n"] * agg["sum_x2"]) - (agg["sum_x"] ** 2)
    can_fit = (agg["n"] >= 2) & (denom != 0)

    agg["slope"] = np.nan
    agg["intercept"] = np.nan
    agg.loc[can_fit, "slope"] = (
        (agg.loc[can_fit, "n"] * agg.loc[can_fit, "sum_xy"])
        - (agg.loc[can_fit, "sum_x"] * agg.loc[can_fit, "sum_y"])
    ) / denom[can_fit]
    agg.loc[can_fit, "intercept"] = (
        agg.loc[can_fit, "sum_y"] - agg.loc[can_fit, "slope"] * agg.loc[can_fit, "sum_x"]
    ) / agg.loc[can_fit, "n"]

    print("Merging trend parameters back onto full dataframe...")
    df = df.merge(
        agg[GROUP_COLS + ["n", "slope", "intercept"]],
        on=GROUP_COLS, how="left"
    )

    df["NDVI_fitted"] = df["intercept"] + df["slope"] * df["Year"]
    df["NDVI_residual"] = np.nan
    has_fit = df["slope"].notna() & valid_mask
    df.loc[has_fit, "NDVI_residual"] = df.loc[has_fit, "NDVI"] - df.loc[has_fit, "NDVI_fitted"]

    print("Z-scoring residuals per group...")
    rgrp = df.groupby(GROUP_COLS, sort=False)["NDVI_residual"]
    df["resid_mean"] = rgrp.transform("mean")   # ignores NaN
    df["resid_std"]  = rgrp.transform("std")     # ignores NaN

    ok_resid = (
        df["NDVI_residual"].notna()
        & (df["n"] >= 2)
        & df["resid_std"].notna()
        & (df["resid_std"] != 0)
    )
    df["NDVI_anomaly_detrended"] = np.nan
    df.loc[ok_resid, "NDVI_anomaly_detrended"] = (
        (df.loc[ok_resid, "NDVI_residual"] - df.loc[ok_resid, "resid_mean"])
        / df.loc[ok_resid, "resid_std"]
    )

    return df

# =========================================================
# Main
# =========================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_data(INPUT_FP)
    df_anom = compute_NDVI_anomaly(df)

    final_csv = df_anom[
        ["Year", "Pentad", "Date", "latitude", "longitude",
         "NDVI", "phase", "phase_name",
         "NDVI_anomaly_detrended", "NDVI_anomaly_nondetrended"]
    ].copy()

    final_csv["Date"] = pd.to_datetime(final_csv["Date"]).dt.strftime("%Y-%m-%d")

    final_csv.to_csv(OUTPUT_FP, index=False)
    print("Done. Final file saved:")
    print(f"  {OUTPUT_FP}")
    print("Columns added: NDVI_anomaly_detrended, NDVI_anomaly_nondetrended")

if __name__ == "__main__":
    main()


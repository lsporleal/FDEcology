# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 13:05:30 2026

@author: 28355914

Computes BOTH:
  - SIF_anomaly_detrended    : linear year-trend (per lat/lon/pentad, across
                                all years) removed first, then residuals z-scored
  - SIF_anomaly_nondetrended : raw SIF z-scored directly per lat/lon/pentad,
                                across all years -- no trend removal
"""

import numpy as np
import pandas as pd

# =========================================================
# Data processing
# =========================================================
def load_data(fp):
    print("Loading full dataset...")
    df = pd.read_csv(fp)
    df["SIF"] = pd.to_numeric(df["GOSIF"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

def compute_SIF_anomaly(df, print_every=1000):
    df = df.copy()

    # create output columns first
    df["SIF_fitted"] = np.nan
    df["SIF_residual"] = np.nan
    df["resid_mean"] = np.nan
    df["resid_std"] = np.nan
    df["SIF_anomaly_detrended"] = np.nan

    df["raw_mean"] = np.nan
    df["raw_std"] = np.nan
    df["SIF_anomaly_nondetrended"] = np.nan

    grouped = df.groupby(["latitude", "longitude", "pentad"], sort=False)
    total_groups = grouped.ngroups

    for i, (_, g) in enumerate(grouped, start=1):
        idx = g.index
        valid = g.dropna(subset=["SIF"])

        if len(valid) >= 2:
            # ---------------------------------------------------
            # Detrended anomaly: remove linear year-trend first,
            # then z-score the residuals
            # ---------------------------------------------------
            slope, intercept = np.polyfit(valid["year"], valid["SIF"], 1)

            fitted = intercept + slope * g["year"]
            residual = g["SIF"] - fitted

            resid_mean = residual.mean()
            resid_std = residual.std()

            df.loc[idx, "SIF_fitted"] = fitted
            df.loc[idx, "SIF_residual"] = residual
            df.loc[idx, "resid_mean"] = resid_mean
            df.loc[idx, "resid_std"] = resid_std

            if pd.notna(resid_std) and resid_std != 0:
                df.loc[idx, "SIF_anomaly_detrended"] = (residual - resid_mean) / resid_std

            # ---------------------------------------------------
            # Non-detrended anomaly: z-score of raw SIF values,
            # no trend removal
            # ---------------------------------------------------
            raw_mean = valid["SIF"].mean()
            raw_std = valid["SIF"].std()

            df.loc[idx, "raw_mean"] = raw_mean
            df.loc[idx, "raw_std"] = raw_std

            if pd.notna(raw_std) and raw_std != 0:
                df.loc[idx, "SIF_anomaly_nondetrended"] = (g["SIF"] - raw_mean) / raw_std

        if i % print_every == 0 or i == total_groups:
            pct = 100 * i / total_groups
            print(f"Processed {i:,}/{total_groups:,} groups ({pct:.2f}%)")

    return df

# =========================================================
# Main
# =========================================================
def main():
    fp = "results/GOSIF_pentad_5day_with_FD_phase.csv"

    df = load_data(fp)
    df_anom = compute_SIF_anomaly(df, print_every=10000)

    final_csv = (
        df_anom[
            ["year", "pentad", "SIF", "date", "latitude", "longitude",
             "phase", "phase_name",
             "SIF_anomaly_detrended", "SIF_anomaly_nondetrended"]
        ]
        .copy()
    )

    final_csv["date"] = pd.to_datetime(final_csv["date"]).dt.strftime("%Y-%m-%d")
    final_csv.to_csv("results/SIF_pentad_5day_with_SIF_anomaly.csv", index=False)

    print("Done. Final file saved:")
    print("SIF_pentad_5day_with_SIF_anomaly.csv")
    print("Columns added: SIF_anomaly_detrended, SIF_anomaly_nondetrended")


if __name__ == "__main__":
    main()
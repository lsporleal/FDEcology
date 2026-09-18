# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 13:02:20 2026

@author: 28355914
"""

"""
GOSIF (8-day at points) -> daily (time interpolation) -> 5-day pentad means (Jan 1 aligned)
-> add Flash Drought phase (0 Normal, 1 Onset, 2 Persistence, 3 CoolDown)
-> write ONE final CSV + optional plots

Inputs:
  - gosif_fp: GOSIF_2000_2015_points_latlon.csv
      required columns: date, GOSIF, latitude, longitude
  - fd_fp: FD_rzsm_3C_1Dr_I_2000_no_relapse_updated.xlsx
      required columns listed in FD_USECOLS below

Outputs:
  - final_out_fp: GOSIF_pentad_5day_with_FD_phase.csv
  - plots_out_dir: sif_phase_plots (PNG panels for a few points)
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch


# ============================================================
# Config
# ============================================================

GOSIF_FILL_VALUES = {32766, 32767}
GOSIF_SCALE = 0.0001

gosif_fp = "data/GOSIF_2000_2015_points_latlon.csv"
fd_fp = "data/FD_rzsm_3C_1Dr_I_2000_no_relapse_updated.xlsx"

final_out_fp = "results/GOSIF_pentad_5day_with_FD_phase.csv"
plots_out_dir = "sif_phase_plots"
os.makedirs("results", exist_ok=True)

START_YEAR = 2000
END_YEAR = 2015

# Plot only these points (optional plotting step)
PLOT_POINTS = [
    (40.147, -102.750),
    (40.147, -103.000),
    (40.147, -103.250),
]

PHASE_NAME_MAP = {0: "Normal", 1: "Onset", 2: "Persistence", 3: "CoolDown"}
PHASE_COLORS = {1: "#fdae61", 2: "#d7191c", 3: "#2c7bb6"}  # onset, persistence, cooldown

CHUNK_SIZE_PLOTS = 1_000_000


# ============================================================
# Small helpers
# ============================================================

def safe_remove(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)


def date_to_year_pentad(d: pd.Timestamp) -> tuple[int, int]:
    """
    Convert date to (year, pentad), aligned to Jan 1.
    Pentad = floor((day_of_year - 1) / 5) + 1.
    Non-leap years can reach 73; leap years can reach 74.
    """
    d = pd.to_datetime(d, errors="coerce")
    if pd.isna(d):
        raise ValueError("date is NaT")
    year = int(d.year)
    doy = int(d.dayofyear)
    pentad = (doy - 1) // 5 + 1
    return year, pentad


def compute_pentad_index(dates: pd.Series) -> pd.Series:
    """Pentad index aligned to Jan 1, 1..74, based on the timestamp itself."""
    y = dates.dt.year
    jan1 = pd.to_datetime(y.astype(str) + "-01-01")
    return ((dates - jan1).dt.days // 5 + 1).astype(int)

# ============================================================
# Build FD phase lookup (fast reindex during GOSIF processing)
# ============================================================

def build_fd_phase_series(
    fd_fp: str,
    year_min: int,
    year_max: int,
    *,
    lat_round: int = 3,
    lon_round: int = 3,
) -> pd.Series:
    """
    Build a sparse lookup:
      index = (latitude, longitude, year, pentad)
      value = phase (int8)

    Priority if overlaps occur:
      Onset (1) > Persistence (2) > CoolDown (3)

    Phase windows:
      Onset:       onset_start_date -> onset_end_date
      Persistence: persistence_start_date -> persistence_end_date
      CoolDown:    cooldown_start_date -> cooldown_end_date
    """
    FD_USECOLS = [
        "latitude", "longitude",
        "onset_start_date",
        "onset_end_date",
        "persistence_start_date",
        "persistence_end_date",
        "cooldown_start_date",
        "cooldown_end_date",
    ]
    
    t0 = time.time()
    fd = pd.read_excel(fd_fp, usecols=FD_USECOLS)

    date_cols = [
        "onset_start_date",
        "onset_end_date",
        "persistence_start_date",
        "persistence_end_date",
        "cooldown_start_date",
        "cooldown_end_date",
    ]
    
    for c in date_cols:
        fd[c] = pd.to_datetime(fd[c], errors="coerce")

    records = []
    
    def add_phase_from_dates(row, start_col, end_col, phase):
        start = row[start_col]
        end = row[end_col]

        if pd.isna(start) or pd.isna(end):
            return

        for d in pd.date_range(start, end, freq="D"):
            y, p = date_to_year_pentad(d)
            if year_min <= y <= year_max:
                records.append((lat, lon, y, p, phase))

    for _, row in fd.iterrows():
        lat = round(float(row["latitude"]), lat_round)
        lon = round(float(row["longitude"]), lon_round)
        
        add_phase_from_dates(row, "onset_start_date", "onset_end_date", 1)
        add_phase_from_dates(row, "persistence_start_date", "persistence_end_date", 2)
        add_phase_from_dates(row, "cooldown_start_date", "cooldown_end_date", 3)

    phase_df = pd.DataFrame(records, columns=["latitude", "longitude", "year", "pentad", "phase"])
    if phase_df.empty:
        empty_idx = pd.MultiIndex.from_arrays([[], [], [], []], names=["latitude", "longitude", "year", "pentad"])
        return pd.Series([], index=empty_idx, dtype=np.int8, name="phase")

    # Resolve overlaps with explicit priority
    priority = {1: 3, 2: 2, 3: 1}  # higher wins
    phase_df["priority"] = phase_df["phase"].map(priority).astype(np.int8)
    phase_df = (
        phase_df.sort_values("priority", ascending=False)
        .drop_duplicates(subset=["latitude", "longitude", "year", "pentad"], keep="first")
        .drop(columns="priority")
    )

    phase_series = (
        phase_df.set_index(["latitude", "longitude", "year", "pentad"])["phase"]
        .astype(np.int8)
        .sort_index()
    )

    print("FD phase lookup keys:", len(phase_series))
    print("FD unique points:", phase_df[["latitude", "longitude"]].drop_duplicates().shape[0])
    print("Elapsed (FD lookup):", (time.time() - t0) / 60, "minutes")
    return phase_series


# ============================================================
# GOSIF processing -> write ONLY final CSV
# ============================================================

def process_gosif_to_pentads_with_phase(
    gosif_fp: str,
    final_out_fp: str,
    phase_series: pd.Series,
    start_year: int,
    end_year: int,
) -> None:
    """
    For each (lat,lon):
      1) sort and de-dup date
      2) daily reindex + time interpolate inside observed range
      3) resample to 5D means aligned to Jan 1
      4) compute (year,pentad)
      5) attach phase via MultiIndex reindex (no giant merge)
      6) append to ONE output CSV
    """
    t0 = time.time()

    daily_grid = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D")
    date_min, date_max = daily_grid.min(), daily_grid.max()

    df = pd.read_csv(gosif_fp)

    # Parse and clean
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["GOSIF"] = pd.to_numeric(df["GOSIF"], errors="coerce")
    df = df.dropna(subset=["date"])

    # Fill values then scale
    df.loc[df["GOSIF"].isin(GOSIF_FILL_VALUES), "GOSIF"] = np.nan
    df["GOSIF"] = df["GOSIF"] * GOSIF_SCALE

    # Time window clip
    df = df[(df["date"] >= date_min) & (df["date"] <= date_max)]
    
    # Keep only GOSIF points that overlap with FD points
    fd_points = pd.read_excel(fd_fp, usecols=["latitude", "longitude"])
    fd_points["latitude"] = pd.to_numeric(fd_points["latitude"], errors="coerce").round(3)
    fd_points["longitude"] = pd.to_numeric(fd_points["longitude"], errors="coerce").round(3)
    fd_points = fd_points.dropna().drop_duplicates()
    
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce").round(3)
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce").round(3)
    
    df = df.merge(fd_points, on=["latitude", "longitude"], how="inner")
    
    print("GOSIF rows after keeping only FD-overlap points:", len(df))
    print("Unique overlap points used:", df[["latitude", "longitude"]].drop_duplicates().shape[0])  

    print("GOSIF rows after cleaning/window:", len(df))

    groups = df.groupby(["latitude", "longitude"], sort=False)
    total_points = len(groups)
    print("Total points:", total_points)

    safe_remove(final_out_fp)
    first_write = True

    for i, ((lat, lon), g) in enumerate(groups, 1):
        # Keep one value per day (if duplicates exist)
        g = g.sort_values("date").drop_duplicates(subset=["date"], keep="last")
        s = g.set_index("date")["GOSIF"]

        # Daily interpolation only inside the observed window
        s_daily = s.reindex(daily_grid).interpolate(method="time", limit_area="inside")

        # Convert daily series to DataFrame
        tmp_daily = s_daily.reset_index()
        tmp_daily.columns = ["date", "GOSIF"]
        
        # Compute year and pentad directly from date
        tmp_daily["year"] = tmp_daily["date"].dt.year
        tmp_daily["pentad"] = compute_pentad_index(tmp_daily["date"])
        
        # Compute pentad means
        s_5d = (
            tmp_daily
            .groupby(["year", "pentad"], as_index=False)["GOSIF"]
            .mean()
        )
        
        # Create representative date for the pentad (Jan1 + 5*(p-1))
        s_5d["date"] = pd.to_datetime(s_5d["year"].astype(str) + "-01-01") + \
                       pd.to_timedelta((s_5d["pentad"] - 1) * 5, unit="D")
        
        # Add coordinates
        s_5d["latitude"] = round(float(lat), 3)
        s_5d["longitude"] = round(float(lon), 3)
        
        tmp = s_5d

        # Phase attach (fast)
        idx = pd.MultiIndex.from_arrays(
            [tmp["latitude"], tmp["longitude"], tmp["year"], tmp["pentad"]],
            names=["latitude", "longitude", "year", "pentad"],
        )
        tmp["phase"] = phase_series.reindex(idx).to_numpy()
        tmp["phase"] = pd.Series(tmp["phase"]).fillna(0).astype(np.int8).to_numpy()
        tmp["phase_name"] = tmp["phase"].map(PHASE_NAME_MAP)

        tmp.to_csv(final_out_fp, mode="w" if first_write else "a", header=first_write, index=False)
        first_write = False

        if (i % 2000 == 0) or (i == total_points):
            elapsed = (time.time() - t0) / 60
            print(f"Processed {i}/{total_points} ({i/total_points*100:.1f}%) | elapsed {elapsed:.2f} min")

    print("\nWrote:", final_out_fp)
    print("Total elapsed:", (time.time() - t0) / 60, "minutes")


# ============================================================
# Plotting (reads final CSV in chunks)
# ============================================================

def contiguous_runs(dates: pd.Series, phases: pd.Series):
    """Return contiguous (start,end,phase) segments for shading."""
    if len(dates) == 0:
        return []
    runs = []
    s = dates.iloc[0]
    p = phases.iloc[0]
    for i in range(1, len(dates)):
        if phases.iloc[i] != p:
            runs.append((s, dates.iloc[i - 1], p))
            s = dates.iloc[i]
            p = phases.iloc[i]
    runs.append((s, dates.iloc[-1], p))
    return runs


def plot_point_year_panels(df_point: pd.DataFrame, lat: float, lon: float, save_path: str):
    """
    df_point required columns: Date (datetime), SIF (float), phase (int)
    Creates multi-year panels (4 columns) with phase shading.
    """
    dfp = df_point.copy()
    dfp["year"] = dfp["Date"].dt.year
    years = sorted(dfp["year"].unique())

    ncols = 4
    nrows = int(np.ceil(len(years) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 2.4 * nrows), sharey=True)
    axes = np.array(axes).reshape(-1)

    for ax_i, year in enumerate(years):
        ax = axes[ax_i]
        d = dfp[dfp["year"] == year].sort_values("Date")

        # Shade phases as spans
        for (s, e, ph) in contiguous_runs(d["Date"], d["phase"]):
            if ph == 0:
                continue
            ax.axvspan(s, e + pd.Timedelta(days=1), color=PHASE_COLORS.get(ph, "#cccccc"), alpha=0.25, lw=0)

        ax.plot(d["Date"], d["SIF"], color="black", linewidth=1.1)
        ax.set_title(str(year), fontsize=10)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.tick_params(axis="x", labelsize=8)

    # Turn off unused axes
    for j in range(len(years), len(axes)):
        axes[j].axis("off")

    legend_elements = [
        Patch(facecolor=PHASE_COLORS[1], alpha=0.25, label="Onset"),
        Patch(facecolor=PHASE_COLORS[2], alpha=0.25, label="Persistence"),
        Patch(facecolor=PHASE_COLORS[3], alpha=0.25, label="CoolDown"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", frameon=False)
    fig.suptitle(f"SIF with Flash Drought Phases\n({lat:.3f}°N, {lon:.3f}°)", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_plots_from_final_csv(final_fp: str, points, out_dir: str):
    """Chunk-read final CSV and plot only selected (lat,lon) points."""
    os.makedirs(out_dir, exist_ok=True)

    pts_df = pd.DataFrame(points, columns=["latitude", "longitude"])
    usecols = ["date", "GOSIF", "latitude", "longitude", "phase"]

    # Collect only selected points in memory (small)
    data_by_point = {pt: [] for pt in points}

    for chunk in pd.read_csv(final_fp, usecols=usecols, chunksize=CHUNK_SIZE_PLOTS):
        sub = chunk.merge(pts_df, on=["latitude", "longitude"], how="inner")
        if sub.empty:
            continue
        for (lat, lon), g in sub.groupby(["latitude", "longitude"], sort=False):
            data_by_point[(float(lat), float(lon))].append(g)

    for (lat, lon) in points:
        parts = data_by_point.get((lat, lon), [])
        if not parts:
            print(f"No data found for point {lat}, {lon}")
            continue

        dfp = pd.concat(parts, ignore_index=True)
        dfp = dfp.rename(columns={"date": "Date", "GOSIF": "SIF"})
        dfp["Date"] = pd.to_datetime(dfp["Date"], errors="coerce")
        dfp = dfp.dropna(subset=["Date"]).sort_values("Date")

        save_path = os.path.join(out_dir, f"SIF_phases_lat_{lat:.3f}_lon_{lon:.3f}.png")
        plot_point_year_panels(dfp[["Date", "SIF", "phase"]], lat, lon, save_path)
        print("Saved:", save_path)


def check_gosif_fd_point_overlap(gosif_fp, fd_fp, lat_round=3, lon_round=3):
    # Read only coordinates
    gosif = pd.read_csv(gosif_fp, usecols=["latitude", "longitude"])
    fd = pd.read_excel(fd_fp, usecols=["latitude", "longitude"])

    # Round same way for matching
    gosif["latitude"] = pd.to_numeric(gosif["latitude"], errors="coerce").round(lat_round)
    gosif["longitude"] = pd.to_numeric(gosif["longitude"], errors="coerce").round(lon_round)

    fd["latitude"] = pd.to_numeric(fd["latitude"], errors="coerce").round(lat_round)
    fd["longitude"] = pd.to_numeric(fd["longitude"], errors="coerce").round(lon_round)

    gosif_points = gosif.dropna().drop_duplicates()
    fd_points = fd.dropna().drop_duplicates()

    common = gosif_points.merge(fd_points, on=["latitude", "longitude"], how="inner")
    gosif_only = gosif_points.merge(fd_points, on=["latitude", "longitude"], how="left", indicator=True)
    gosif_only = gosif_only[gosif_only["_merge"] == "left_only"].drop(columns="_merge")

    fd_only = fd_points.merge(gosif_points, on=["latitude", "longitude"], how="left", indicator=True)
    fd_only = fd_only[fd_only["_merge"] == "left_only"].drop(columns="_merge")

    print("\n===== GOSIF vs FD point overlap =====")
    print("Unique GOSIF points:", len(gosif_points))
    print("Unique FD points:", len(fd_points))
    print("Common points:", len(common))
    print("GOSIF only:", len(gosif_only))
    print("FD only:", len(fd_only))

    return common, gosif_only, fd_only

# ============================================================
# Main
# ============================================================

def main():
    
    # 0) Check coordinate overlap first
    common, gosif_only, fd_only = check_gosif_fd_point_overlap(
        gosif_fp=gosif_fp,
        fd_fp=fd_fp,
        lat_round=3,
        lon_round=3
    )

    # 1) Build FD phase lookup (sparse Series)
    phase_series = build_fd_phase_series(fd_fp, START_YEAR, END_YEAR)

    # 2) Process GOSIF and write only the final merged CSV
    process_gosif_to_pentads_with_phase(
        gosif_fp=gosif_fp,
        final_out_fp=final_out_fp,
        phase_series=phase_series,
        start_year=START_YEAR,
        end_year=END_YEAR,
    )

    # 3) Optional plots from the final CSV (comment out if not needed)
    make_plots_from_final_csv(final_out_fp, PLOT_POINTS, plots_out_dir)


if __name__ == "__main__":
    main()
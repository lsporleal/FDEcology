#!/usr/bin/env python3
"""
Flash-drought processing workflow.

Execution order:
1. Daily gridded CSV files -> 5-day pentad mean arrays.
2. Pentad mean arrays -> pentad-of-year empirical percentile arrays.
3. Pentad percentile arrays -> flash-drought event catalogue.


"""

import os
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def _resolved_path(env_name: str, default: Path) -> Path:
    return Path(os.environ.get(env_name, str(default))).expanduser().resolve()


PROJECT_ROOT = _resolved_path(
    "FD_PROJECT_ROOT",
    Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd(),
)
DAILY_INPUT_DIR = _resolved_path("FD_DAILY_INPUT_DIR", PROJECT_ROOT / "data" / "daily")
OUTPUT_ROOT = _resolved_path("FD_OUTPUT_ROOT", PROJECT_ROOT / "results")

PENTAD_MEAN_DIR = OUTPUT_ROOT / "pentad_means"
PERCENTILE_DIR = OUTPUT_ROOT / "pentad_percentiles"
EVENT_DIR = OUTPUT_ROOT / "flash_drought_events"

for directory in (PENTAD_MEAN_DIR, PERCENTILE_DIR, EVENT_DIR):
    directory.mkdir(parents=True, exist_ok=True)


PENTAD_CFG = {
    "daily_input_dir": DAILY_INPUT_DIR,
    "pentad_output_dir": PENTAD_MEAN_DIR,
    "input_csv_template": os.environ.get(
        "FD_DAILY_CSV_TEMPLATE",
        "{year}_{region_tag}.csv",
    ),
    "pentad_mean_file_template": "pentad_mean_{year}_{region_tag}.npy",
    "pentad_meta_file_template": "pentad_mean_{year}_{region_tag}_meta.npz",
    "region_tag": os.environ.get("FD_REGION_TAG", "USA"),
    "years": list(range(2000, 2025)),
    "pentad_length_days": 5,
    "enforce_fixed_pentad_count": True,
    "fixed_pentads_per_year": 73,
    "drop_extra_end_days": True,
    "allow_partial_final_pentad": False,
    "latitude_col": "latitude",
    "longitude_col": "longitude",
    "date_start_col_idx": 2,
    "overwrite": True,
    "dtype": np.float32,
}


def daily_input_csv_path(year: int) -> Path:
    return PENTAD_CFG["daily_input_dir"] / PENTAD_CFG["input_csv_template"].format(
        year=year,
        region_tag=PENTAD_CFG["region_tag"],
    )


def pentad_mean_output_paths(year: int):
    data_path = PENTAD_CFG["pentad_output_dir"] / PENTAD_CFG["pentad_mean_file_template"].format(
        year=year,
        region_tag=PENTAD_CFG["region_tag"],
    )
    meta_path = PENTAD_CFG["pentad_output_dir"] / PENTAD_CFG["pentad_meta_file_template"].format(
        year=year,
        region_tag=PENTAD_CFG["region_tag"],
    )
    return data_path, meta_path


def compute_pentad_windows(n_days: int):
    pentad_len = PENTAD_CFG["pentad_length_days"]

    if PENTAD_CFG["enforce_fixed_pentad_count"]:
        target_days = PENTAD_CFG["fixed_pentads_per_year"] * pentad_len

        if n_days < target_days:
            raise ValueError(
                f"Input has only {n_days} days, but fixed mode requires at least {target_days} days."
            )

        if n_days > target_days and not PENTAD_CFG["drop_extra_end_days"]:
            raise ValueError(
                f"Input has {n_days} days > {target_days}, but drop_extra_end_days=False."
            )

        n_days_used = target_days
        n_days_dropped = n_days - n_days_used
        windows = [
            (i * pentad_len, (i + 1) * pentad_len)
            for i in range(PENTAD_CFG["fixed_pentads_per_year"])
        ]
        return windows, n_days_used, n_days_dropped

    if PENTAD_CFG["allow_partial_final_pentad"]:
        n_pentads = (n_days + pentad_len - 1) // pentad_len
    else:
        n_pentads = n_days // pentad_len

    n_days_used = (
        min(n_days, n_pentads * pentad_len)
        if not PENTAD_CFG["allow_partial_final_pentad"]
        else n_days
    )
    n_days_dropped = n_days - n_days_used

    windows = []
    for p in range(n_pentads):
        start_idx = p * pentad_len
        end_idx = min(start_idx + pentad_len, n_days)
        if end_idx > start_idx:
            windows.append((start_idx, end_idx))

    return windows, n_days_used, n_days_dropped


def process_daily_year(year: int):
    input_path = daily_input_csv_path(year)
    out_data_path, out_meta_path = pentad_mean_output_paths(year)

    if not input_path.exists():
        print(f"[SKIP] Missing input: {input_path}")
        return None

    if (
        not PENTAD_CFG["overwrite"]
        and out_data_path.exists()
        and out_meta_path.exists()
    ):
        print(f"[SKIP] Output exists: {out_data_path.name}")
        meta = np.load(out_meta_path, allow_pickle=True)
        return {
            "year": year,
            "n_points": int(meta["n_points"]),
            "n_pentads": int(meta["n_pentads"]),
            "n_days_original": int(meta["n_days_original"]),
            "n_days_used": int(meta["n_days_used"]),
            "n_days_dropped": int(meta["n_days_dropped"]),
            "status": "existing",
        }

    print(f"\n[INFO] Processing year {year}")
    df = pd.read_csv(input_path, low_memory=False)

    lat_col = PENTAD_CFG["latitude_col"]
    lon_col = PENTAD_CFG["longitude_col"]

    if lat_col not in df.columns or lon_col not in df.columns:
        raise KeyError(
            f"Expected coordinate columns '{lat_col}' and '{lon_col}' in {input_path}"
        )

    lats = df[lat_col].to_numpy(dtype=PENTAD_CFG["dtype"])
    lons = df[lon_col].to_numpy(dtype=PENTAD_CFG["dtype"])

    date_cols = list(df.columns[PENTAD_CFG["date_start_col_idx"] :])
    if len(date_cols) == 0:
        raise ValueError(f"No daily data columns found in {input_path}")

    try:
        parsed_dates = pd.to_datetime(date_cols)
    except Exception as exc:
        raise ValueError(f"Could not parse date columns in {input_path}: {exc}") from exc

    data = df.iloc[:, PENTAD_CFG["date_start_col_idx"] :].to_numpy(
        dtype=PENTAD_CFG["dtype"]
    )
    n_points, n_days = data.shape
    print(f"[INFO] Grid points: {n_points:,} | Daily columns: {n_days}")

    sort_idx = np.argsort(parsed_dates)
    parsed_dates = parsed_dates[sort_idx]
    data = data[:, sort_idx]

    windows, n_days_used, n_days_dropped = compute_pentad_windows(n_days=n_days)

    if n_days_dropped > 0:
        warnings.warn(
            f"{year}: dropping {n_days_dropped} trailing day(s) to keep a consistent pentad calendar."
        )

    pentad_data = np.full(
        (n_points, len(windows)),
        np.nan,
        dtype=PENTAD_CFG["dtype"],
    )
    pentad_start_dates = []
    pentad_end_dates = []

    for p, (start_idx, end_idx) in enumerate(windows):
        slice_data = data[:, start_idx:end_idx]
        pentad_data[:, p] = np.nanmean(slice_data, axis=1)
        pentad_start_dates.append(parsed_dates[start_idx])
        pentad_end_dates.append(parsed_dates[end_idx - 1])

    np.save(out_data_path, pentad_data)
    np.savez(
        out_meta_path,
        year=year,
        region_tag=PENTAD_CFG["region_tag"],
        n_points=n_points,
        n_pentads=len(windows),
        n_days_original=n_days,
        n_days_used=n_days_used,
        n_days_dropped=n_days_dropped,
        pentad_length_days=PENTAD_CFG["pentad_length_days"],
        latitude=lats,
        longitude=lons,
        pentad_start_dates=np.array(pentad_start_dates, dtype="datetime64[D]"),
        pentad_end_dates=np.array(pentad_end_dates, dtype="datetime64[D]"),
        source_file=str(input_path),
    )

    print(f"[OK] Saved data: {out_data_path.name}")
    print(f"[OK] Saved meta: {out_meta_path.name}")
    print(
        f"[OK] Shape: {pentad_data.shape[0]:,} points x "
        f"{pentad_data.shape[1]} pentads"
    )

    return {
        "year": year,
        "n_points": n_points,
        "n_pentads": len(windows),
        "n_days_original": n_days,
        "n_days_used": n_days_used,
        "n_days_dropped": n_days_dropped,
        "status": "processed",
    }


def run_daily_to_pentad():
    print("=== DAILY CSV -> PENTAD MEAN ARRAYS ===")
    print(f"Input dir : {PENTAD_CFG['daily_input_dir']}")
    print(f"Output dir: {PENTAD_CFG['pentad_output_dir']}")
    print(f"Years     : {PENTAD_CFG['years'][0]} .. {PENTAD_CFG['years'][-1]}")
    print(f"Pentad len: {PENTAD_CFG['pentad_length_days']} days")
    print(f"Fixed 73  : {PENTAD_CFG['enforce_fixed_pentad_count']}\n")

    results = []
    for year in tqdm(PENTAD_CFG["years"], desc="Years"):
        try:
            result = process_daily_year(year)
            if result is not None:
                results.append(result)
        except Exception as exc:
            print(f"[ERROR] Year {year}: {exc}")

    summary_df = pd.DataFrame(results)
    print("\n=== SUMMARY ===")
    if summary_df.empty:
        print("No annual pentad files were processed.")
    else:
        print(summary_df.to_string(index=False))


PERCENTILE_CFG = {
    "pentad_mean_input_dir": PENTAD_MEAN_DIR,
    "percentile_output_dir": PERCENTILE_DIR,
    "pentad_mean_file_template": "pentad_mean_{year}_{region_tag}.npy",
    "pentad_mean_meta_file_template": "pentad_mean_{year}_{region_tag}_meta.npz",
    "percentile_file_template": "pentad_percentiles_{year}_{region_tag}.npy",
    "percentile_meta_file_template": "pentad_percentiles_{year}_{region_tag}_meta.npz",
    "region_tag": os.environ.get("FD_REGION_TAG", "USA"),
    "years": list(range(2000, 2025)),
    "expected_pentads_per_year": 73,
    "rank_method": "average",
    "na_option": "keep",
    "percentile_scale": 100.0,
    "include_target_year_in_climatology": True,
    "chunk_size_points": 25000,
    "dtype": np.float32,
    "overwrite": True,
    "require_same_n_points": True,
    "require_same_lat_lon": False,
}

def mean_file_path(year: int) -> Path:
    return PERCENTILE_CFG["pentad_mean_input_dir"] / PERCENTILE_CFG["pentad_mean_file_template"].format(
        year=year,
        region_tag=PERCENTILE_CFG["region_tag"],
    )

def mean_meta_path(year: int) -> Path:
    return PERCENTILE_CFG["pentad_mean_input_dir"] / PERCENTILE_CFG["pentad_mean_meta_file_template"].format(
        year=year,
        region_tag=PERCENTILE_CFG["region_tag"],
    )

def percentile_output_file_path(year: int) -> Path:
    return PERCENTILE_CFG["percentile_output_dir"] / PERCENTILE_CFG["percentile_file_template"].format(
        year=year,
        region_tag=PERCENTILE_CFG["region_tag"],
    )

def percentile_output_meta_path(year: int) -> Path:
    return PERCENTILE_CFG["percentile_output_dir"] / PERCENTILE_CFG["percentile_meta_file_template"].format(
        year=year,
        region_tag=PERCENTILE_CFG["region_tag"],
    )

def load_available_pentad_mean_files():
    """
    Returns
    -------
    years_loaded : list[int]
    arr_map      : dict[year -> np.memmap]
    meta_map     : dict[year -> np.lib.npyio.NpzFile or None]
    """
    years_loaded = []
    arr_map = {}
    meta_map = {}

    ref_n_points = None
    ref_n_pentads = None
    ref_lat = None
    ref_lon = None

    for year in tqdm(PERCENTILE_CFG["years"], desc="Scanning pentad mean files"):
        data_path = mean_file_path(year)
        meta_path = mean_meta_path(year)

        if not data_path.exists():
            warnings.warn(f"Missing pentad mean file: {data_path}")
            continue

        arr = np.load(data_path, mmap_mode="r")
        if arr.ndim != 2:
            raise ValueError(f"{data_path} must be 2D, got shape {arr.shape}")

        n_points, n_pentads = arr.shape

        if n_pentads != PERCENTILE_CFG["expected_pentads_per_year"]:
            raise ValueError(
                f"{data_path} has {n_pentads} pentads, expected {PERCENTILE_CFG['expected_pentads_per_year']}. "
                f"Make sure Script 1 used a fixed 73-pentad year."
            )

        meta = np.load(meta_path, allow_pickle=True) if meta_path.exists() else None

        if ref_n_points is None:
            ref_n_points = n_points
            ref_n_pentads = n_pentads
            if meta is not None and ("latitude" in meta.files) and ("longitude" in meta.files):
                ref_lat = meta["latitude"]
                ref_lon = meta["longitude"]
        else:
            if PERCENTILE_CFG["require_same_n_points"] and n_points != ref_n_points:
                raise ValueError(
                    f"Inconsistent n_points. {data_path} has {n_points}, expected {ref_n_points}."
                )

            if n_pentads != ref_n_pentads:
                raise ValueError(
                    f"Inconsistent n_pentads. {data_path} has {n_pentads}, expected {ref_n_pentads}."
                )

            if PERCENTILE_CFG["require_same_lat_lon"] and meta is not None and ref_lat is not None and ref_lon is not None:
                if ("latitude" in meta.files) and ("longitude" in meta.files):
                    same_lat = np.array_equal(meta["latitude"], ref_lat)
                    same_lon = np.array_equal(meta["longitude"], ref_lon)
                    if not (same_lat and same_lon):
                        raise ValueError(f"Coordinate mismatch detected in {meta_path}")

        years_loaded.append(year)
        arr_map[year] = arr
        meta_map[year] = meta

    if not years_loaded:
        raise FileNotFoundError(
            "No pentad mean files found. Run Script 1 first or update the input directory/template."
        )

    return years_loaded, arr_map, meta_map

def create_output_memmaps(years_loaded, n_points, n_pentads):
    """
    Creates on-disk .npy memmaps for percentile arrays so large datasets
    can be written chunk-by-chunk without holding everything in RAM.
    """
    out_map = {}
    for year in years_loaded:
        out_path = percentile_output_file_path(year)

        if out_path.exists() and (not PERCENTILE_CFG["overwrite"]):
            raise FileExistsError(
                f"Output exists and overwrite=False: {out_path}"
            )

        mm = np.lib.format.open_memmap(
            out_path,
            mode="w+",
            dtype=PERCENTILE_CFG["dtype"],
            shape=(n_points, n_pentads),
        )
        mm[:] = np.nan
        out_map[year] = mm

    return out_map

def compute_chunk_percentiles(block_2d: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    block_2d : ndarray of shape (n_years, n_points_in_chunk)
        Values for a single pentad across years and points.

    Returns
    -------
    pct_2d : ndarray of shape (n_years, n_points_in_chunk)
        Empirical percentiles (0-100) ranked down the year dimension
        independently for each point/column.
    """
    df = pd.DataFrame(block_2d)
    pct = df.rank(
        axis=0,
        method=PERCENTILE_CFG["rank_method"],
        pct=True,
        na_option=PERCENTILE_CFG["na_option"],
    ).to_numpy(dtype=PERCENTILE_CFG["dtype"])

    pct *= PERCENTILE_CFG["percentile_scale"]
    return pct

def run_percentile_stage():
    print("=== LOAD PENTAD MEAN FILES ===")
    years_loaded, arr_map, meta_map = load_available_pentad_mean_files()

    ref_year = years_loaded[0]
    n_points, n_pentads = arr_map[ref_year].shape

    print(f"[OK] Loaded years         : {years_loaded[0]} .. {years_loaded[-1]}")
    print(f"[OK] Number of years     : {len(years_loaded)}")
    print(f"[OK] Grid points         : {n_points:,}")
    print(f"[OK] Pentads per year    : {n_pentads}")
    print(f"[OK] Output percentile dir: {PERCENTILE_CFG['percentile_output_dir']}")

    print("\n=== CREATE OUTPUT PERCENTILE FILES ===")
    out_map = create_output_memmaps(years_loaded, n_points, n_pentads)

    print("\n=== COMPUTE PERCENTILES ===")
    print("Method: for each pentad-of-year, rank values across all loaded years at each grid point.")

    for start in tqdm(range(0, n_points, PERCENTILE_CFG["chunk_size_points"]), desc="Point chunks"):
        end = min(start + PERCENTILE_CFG["chunk_size_points"], n_points)

        for p in range(n_pentads):
            block = np.stack(
                [arr_map[year][start:end, p] for year in years_loaded],
                axis=0,
            ).astype(PERCENTILE_CFG["dtype"], copy=False)

            pct_block = compute_chunk_percentiles(block)

            for yi, year in enumerate(years_loaded):
                out_map[year][start:end, p] = pct_block[yi, :]

    for year in years_loaded:
        out_map[year].flush()

    print("[OK] Percentile arrays written to disk.")

    print("\n=== SAVE METADATA ===")

    for year in tqdm(years_loaded, desc="Saving meta files"):
        src_meta = meta_map[year]
        meta_out = {
            "year": year,
            "region_tag": PERCENTILE_CFG["region_tag"],
            "n_points": n_points,
            "n_pentads": n_pentads,
            "years_used_for_climatology": np.array(years_loaded, dtype=np.int32),
            "climatology_n_years": len(years_loaded),
            "percentile_definition": "empirical rank percentile by pentad-of-year across loaded years",
            "rank_method": PERCENTILE_CFG["rank_method"],
            "na_option": PERCENTILE_CFG["na_option"],
            "percentile_scale": PERCENTILE_CFG["percentile_scale"],
            "include_target_year_in_climatology": PERCENTILE_CFG["include_target_year_in_climatology"],
            "source_pentad_mean_file": str(mean_file_path(year)),
            "source_pentad_mean_meta_file": str(mean_meta_path(year)),
        }

        if src_meta is not None:
            for key in [
                "latitude",
                "longitude",
                "pentad_start_dates",
                "pentad_end_dates",
                "pentad_length_days",
                "n_days_original",
                "n_days_used",
                "n_days_dropped",
                "source_file",
            ]:
                if key in src_meta.files:
                    meta_out[key] = src_meta[key]

        np.savez(percentile_output_meta_path(year), **meta_out)

    print("[OK] Percentile metadata files saved.")

    summary_rows = []
    for year in years_loaded:
        out_path = percentile_output_file_path(year)
        meta_path = percentile_output_meta_path(year)
        summary_rows.append({
            "year": year,
            "percentile_file": str(out_path),
            "meta_file": str(meta_path),
            "shape": tuple(np.load(out_path, mmap_mode="r").shape),
        })

    summary_df = pd.DataFrame(summary_rows)
    print("\n=== SUMMARY ===")
    print(summary_df.to_string(index=False))

    print("\nDone.")


EVENT_CFG = {
    "percentile_input_dir": PERCENTILE_DIR,
    "event_output_dir": EVENT_DIR,
    "percentile_file_template": "pentad_percentiles_{year}_{region_tag}.npy",
    "percentile_meta_file_template": "pentad_percentiles_{year}_{region_tag}_meta.npz",
    "output_csv_name": "flash_drought_events.csv",
    "backup_prefix": "flash_drought_events_backup",
    "region_tag": os.environ.get("FD_REGION_TAG", "USA"),
    "years": list(range(2000, 2025)),
    "max_pentads_per_year": 73,
    "start_threshold": 40.0,
    "drought_threshold": 20.0,
    "min_drop_rate": 5.0,
    "max_onset_pentads": 4,
    "min_event_length": 4,
    "min_below20_pentads": 1,
    "full_recovery_threshold": 40.0,
    "start_is_geq": True,
    "drought_is_strict_lt": True,
    "recovery20_is_strict_gt": True,
    "recovery40_is_strict_gt": True,
    "require_non_increasing_onset": False,
    "post_event_skip_pentads": 3,
    "save_phase_milestones": True,
    "onset_milestone_thresholds": [30, 20],
    "post20_drop_thresholds": [10, 5, 0],
    "recovery_milestone_thresholds": [20, 30, 40],
    "attach_lat_lon": True,
    "overwrite_output_csv": True,
}

def percentile_input_file_path(year: int) -> Path:
    return EVENT_CFG["percentile_input_dir"] / EVENT_CFG["percentile_file_template"].format(
        year=year,
        region_tag=EVENT_CFG["region_tag"],
    )

def percentile_input_meta_path(year: int) -> Path:
    return EVENT_CFG["percentile_input_dir"] / EVENT_CFG["percentile_meta_file_template"].format(
        year=year,
        region_tag=EVENT_CFG["region_tag"],
    )

def idx_to_year_pentad(idx: int, years_loaded: list[int], n_pentads_year: int):
    year = years_loaded[idx // n_pentads_year]
    pentad = (idx % n_pentads_year) + 1
    return int(year), int(pentad)

def seq_idx_to_onebased_global_pentad(idx: int) -> int:
    return int(idx) + 1

def year_pentad_code(year: int, pentad: int, n_pentads_year: int) -> int:
    return int(year) * n_pentads_year + int(pentad)

def add_milestone_cols(d: dict, prefix: str, idx: int | None, years_loaded: list[int], n_pentads_year: int):
    if idx is None:
        d[f"{prefix}_year"] = np.nan
        d[f"{prefix}_pentad"] = np.nan
        d[f"{prefix}_seq_idx"] = np.nan
        d[f"{prefix}_code"] = np.nan
        return

    y, p = idx_to_year_pentad(idx, years_loaded, n_pentads_year)
    d[f"{prefix}_year"] = y
    d[f"{prefix}_pentad"] = p
    d[f"{prefix}_seq_idx"] = seq_idx_to_onebased_global_pentad(idx)
    d[f"{prefix}_code"] = year_pentad_code(y, p, n_pentads_year)

def first_idx_leq(ts: np.ndarray, start: int, end: int, thr: float):
    if start > end:
        return None
    seg = ts[start:end + 1]
    m = np.where(~np.isnan(seg) & (seg <= thr))[0]
    return (start + int(m[0])) if m.size else None

def first_idx_geq(ts: np.ndarray, start: int, end: int, thr: float):
    if start > end:
        return None
    seg = ts[start:end + 1]
    m = np.where(~np.isnan(seg) & (seg >= thr))[0]
    return (start + int(m[0])) if m.size else None

def first_idx_gt(ts: np.ndarray, start: int, end: int, thr: float):
    if start > end:
        return None
    seg = ts[start:end + 1]
    m = np.where(~np.isnan(seg) & (seg > thr))[0]
    return (start + int(m[0])) if m.size else None

def nanargmin_in_window(ts: np.ndarray, start: int, end: int):
    if start > end:
        return None
    seg = ts[start:end + 1]
    if np.all(np.isnan(seg)):
        return None
    return start + int(np.nanargmin(seg))

def start_ok(value: float) -> bool:
    if np.isnan(value):
        return False
    return (value >= EVENT_CFG["start_threshold"]) if EVENT_CFG["start_is_geq"] else (value > EVENT_CFG["start_threshold"])

def drought_ok(value: float) -> bool:
    if np.isnan(value):
        return False
    return (value < EVENT_CFG["drought_threshold"]) if EVENT_CFG["drought_is_strict_lt"] else (value <= EVENT_CFG["drought_threshold"])

def recovery20_ok(value: float) -> bool:
    if np.isnan(value):
        return False
    return (value > EVENT_CFG["drought_threshold"]) if EVENT_CFG["recovery20_is_strict_gt"] else (value >= EVENT_CFG["drought_threshold"])

def recovery40_ok(value: float) -> bool:
    if np.isnan(value):
        return False
    return (value > EVENT_CFG["full_recovery_threshold"]) if EVENT_CFG["recovery40_is_strict_gt"] else (value >= EVENT_CFG["full_recovery_threshold"])

def onset_path_ok(path: np.ndarray) -> bool:
    if not EVENT_CFG["require_non_increasing_onset"]:
        return True
    if np.isnan(path).any():
        return False
    return bool(np.all(np.diff(path) <= 0.0))

def run_event_stage():
    print("=== LOAD PENTAD PERCENTILES ===")

    all_percentiles = []
    years_loaded = []
    meta_ref = None

    for year in tqdm(EVENT_CFG["years"], desc="Loading percentile files"):
        f = percentile_input_file_path(year)
        if not f.exists():
            warnings.warn(f"Missing percentile file: {f}")
            continue

        arr = np.load(f)
        if arr.ndim != 2:
            raise ValueError(f"{f} must be 2D, got shape {arr.shape}")

        if arr.shape[1] != EVENT_CFG["max_pentads_per_year"]:
            raise ValueError(
                f"{f} has shape {arr.shape}; expected second dimension = {EVENT_CFG['max_pentads_per_year']}"
            )

        all_percentiles.append(arr)
        years_loaded.append(year)

        if meta_ref is None:
            mf = percentile_input_meta_path(year)
            if mf.exists():
                meta_ref = np.load(mf, allow_pickle=True)

    if not all_percentiles:
        raise FileNotFoundError("No percentile files found. Update percentile_input_dir and file template.")

    percentile_stack = np.stack(all_percentiles, axis=2)
    n_points, n_pentads_year, n_years = percentile_stack.shape

    print(f"[OK] Loaded {n_points:,} grid points x {n_pentads_year} pentads/year x {n_years} years")
    print(f"[OK] Years loaded: {years_loaded[0]} .. {years_loaded[-1]}")

    print("\n=== DETECT FLASH DROUGHT EVENTS ===")
    events = []

    for point_idx in tqdm(range(n_points), desc="Grid points"):
        ts = percentile_stack[point_idx, :, :].flatten(order="F")
        n_total = len(ts)
        t = 0

        while t < n_total - 1:
            if not start_ok(ts[t]):
                t += 1
                continue

            candidate_start = t
            candidate_onset_end = None
            candidate_onset_rate = None

            for ahead in range(1, EVENT_CFG["max_onset_pentads"] + 1):
                j = candidate_start + ahead
                if j >= n_total:
                    break
                if np.isnan(ts[j]):
                    break

                path = ts[candidate_start:j + 1]
                if not onset_path_ok(path):
                    break

                current_drop = ts[candidate_start] - ts[j]
                current_rate = current_drop / ahead

                if drought_ok(ts[j]) and current_rate >= EVENT_CFG["min_drop_rate"]:
                    candidate_onset_end = j
                    candidate_onset_rate = float(current_rate)
                    break

            if candidate_onset_end is None:
                t += 1
                continue

            k = candidate_start

            below20_count = 0
            relapse_count = 0
            has_relapse = False

            first_recovery20 = None
            final_recovery20 = None
            full_recovery40 = None

            min_value = np.inf
            min_index = None

            state = "pre_recovery20"   # pre_recovery20 | between20and40 | fully_recovered
            seen_recovery20_once = False

            while k < n_total:
                value = ts[k]

                if np.isnan(value):
                    k += 1
                    continue

                if value < min_value:
                    min_value = float(value)
                    min_index = k

                if drought_ok(value):
                    below20_count += 1

                    if seen_recovery20_once and full_recovery40 is None and state == "between20and40":
                        relapse_count += 1
                        has_relapse = True
                        state = "pre_recovery20"

                if recovery20_ok(value) and k > candidate_onset_end:
                    if first_recovery20 is None:
                        first_recovery20 = k
                        seen_recovery20_once = True
                        state = "between20and40"

                    final_recovery20 = k
                    if full_recovery40 is None:
                        state = "between20and40"

                if recovery40_ok(value) and k > candidate_onset_end:
                    full_recovery40 = k
                    state = "fully_recovered"
                    break

                k += 1

            if first_recovery20 is None:
                censored_recovery20 = True
                final_recovery20 = n_total - 1
            else:
                censored_recovery20 = False

            if full_recovery40 is None:
                censored_recovery40 = True
            else:
                censored_recovery40 = False

            below40_length = int(final_recovery20 - candidate_start + 1)
            fd_total_length = int(final_recovery20 - candidate_start + 1)

            if full_recovery40 is not None:
                full_episode_length = int(full_recovery40 - candidate_start + 1)
                recovery_tail_length = int(full_recovery40 - final_recovery20 + 1)
            else:
                full_episode_length = np.nan
                recovery_tail_length = np.nan

            if fd_total_length < EVENT_CFG["min_event_length"]:
                t += 1
                continue

            if below20_count < EVENT_CFG["min_below20_pentads"]:
                t += 1
                continue

            start_year, start_pentad = idx_to_year_pentad(candidate_start, years_loaded, n_pentads_year)
            onset_end_year, onset_end_pentad = idx_to_year_pentad(candidate_onset_end, years_loaded, n_pentads_year)
            min_year, min_pentad = idx_to_year_pentad(min_index, years_loaded, n_pentads_year)

            rec = {
                "event_id": len(events) + 1,
                "point_idx": int(point_idx),

                "start_year": start_year,
                "start_pentad": start_pentad,
                "start_percentile": float(ts[candidate_start]),

                "onset_end_year": onset_end_year,
                "onset_end_pentad": onset_end_pentad,
                "onset_end_percentile": float(ts[candidate_onset_end]),
                "onset_length_pentads": int(candidate_onset_end - candidate_start),
                "onset_drop_rate": float(candidate_onset_rate),

                "below40_length_pentads": below40_length,
                "below20_length_pentads": int(below20_count),

                "first_recovery20_year": np.nan,
                "first_recovery20_pentad": np.nan,
                "first_recovery20_index": np.nan,
                "first_recovery20_seq_idx": np.nan,
                "first_recovery20_code": np.nan,

                "final_recovery20_year": np.nan,
                "final_recovery20_pentad": np.nan,
                "final_recovery20_index": np.nan,
                "final_recovery20_seq_idx": np.nan,
                "final_recovery20_code": np.nan,

                "full_recovery40_year": np.nan,
                "full_recovery40_pentad": np.nan,
                "full_recovery40_index": np.nan,
                "full_recovery40_seq_idx": np.nan,
                "full_recovery40_code": np.nan,

                "relapse_count": int(relapse_count),
                "has_relapse": bool(has_relapse),

                "min_percentile": float(min_value) if np.isfinite(min_value) else np.nan,
                "min_year": min_year,
                "min_pentad": min_pentad,
                "min_index": int(min_index) if min_index is not None else np.nan,
                "min_seq_idx": seq_idx_to_onebased_global_pentad(min_index) if min_index is not None else np.nan,
                "min_code": year_pentad_code(min_year, min_pentad, n_pentads_year) if min_index is not None else np.nan,

                "drop_magnitude": float(ts[candidate_start] - min_value) if np.isfinite(min_value) else np.nan,

                "fd_total_length_pentads": fd_total_length,
                "full_episode_length_pentads": full_episode_length,
                "recovery_tail_length": recovery_tail_length,

                "censored_recovery20": bool(censored_recovery20),
                "censored_recovery40": bool(censored_recovery40),

                "recovery_year": np.nan,
                "recovery_pentad": np.nan,
                "censored": bool(censored_recovery20),
                "duration_pentads": fd_total_length,
                "event_total_duration_pentads": fd_total_length,
                "min_percentile_during_event": float(min_value) if np.isfinite(min_value) else np.nan,
                "onset_rate_pct_per_pentad": float(candidate_onset_rate),
            }

            rec["start_index"] = int(candidate_start)
            rec["start_seq_idx"] = seq_idx_to_onebased_global_pentad(candidate_start)
            rec["start_code"] = year_pentad_code(start_year, start_pentad, n_pentads_year)

            rec["onset_end_index"] = int(candidate_onset_end)
            rec["onset_end_seq_idx"] = seq_idx_to_onebased_global_pentad(candidate_onset_end)
            rec["onset_end_code"] = year_pentad_code(onset_end_year, onset_end_pentad, n_pentads_year)

            if first_recovery20 is not None:
                y, p = idx_to_year_pentad(first_recovery20, years_loaded, n_pentads_year)
                rec["first_recovery20_year"] = y
                rec["first_recovery20_pentad"] = p
                rec["first_recovery20_index"] = int(first_recovery20)
                rec["first_recovery20_seq_idx"] = seq_idx_to_onebased_global_pentad(first_recovery20)
                rec["first_recovery20_code"] = year_pentad_code(y, p, n_pentads_year)

            if final_recovery20 is not None:
                y, p = idx_to_year_pentad(final_recovery20, years_loaded, n_pentads_year)
                rec["final_recovery20_year"] = y
                rec["final_recovery20_pentad"] = p
                rec["final_recovery20_index"] = int(final_recovery20)
                rec["final_recovery20_seq_idx"] = seq_idx_to_onebased_global_pentad(final_recovery20)
                rec["final_recovery20_code"] = year_pentad_code(y, p, n_pentads_year)

                rec["recovery_year"] = y
                rec["recovery_pentad"] = p
                rec["fd_recovery_seq_idx"] = seq_idx_to_onebased_global_pentad(final_recovery20)
                rec["fd_recovery_code"] = year_pentad_code(y, p, n_pentads_year)
            else:
                rec["fd_recovery_seq_idx"] = np.nan
                rec["fd_recovery_code"] = np.nan

            if full_recovery40 is not None:
                y, p = idx_to_year_pentad(full_recovery40, years_loaded, n_pentads_year)
                rec["full_recovery40_year"] = y
                rec["full_recovery40_pentad"] = p
                rec["full_recovery40_index"] = int(full_recovery40)
                rec["full_recovery40_seq_idx"] = seq_idx_to_onebased_global_pentad(full_recovery40)
                rec["full_recovery40_code"] = year_pentad_code(y, p, n_pentads_year)

            add_milestone_cols(rec, "fd_min", min_index, years_loaded, n_pentads_year)

            if EVENT_CFG["save_phase_milestones"]:
                for thr in EVENT_CFG["onset_milestone_thresholds"]:
                    if thr == EVENT_CFG["drought_threshold"]:
                        idx_thr = candidate_onset_end
                    else:
                        idx_thr = first_idx_leq(ts, candidate_start, candidate_onset_end, thr)
                    add_milestone_cols(rec, f"fd_drop_to_{int(thr)}", idx_thr, years_loaded, n_pentads_year)

                if min_index is not None:
                    for thr in EVENT_CFG["post20_drop_thresholds"]:
                        idx_thr = first_idx_leq(ts, candidate_onset_end, min_index, thr)
                        add_milestone_cols(rec, f"fd_drop_to_{int(thr)}", idx_thr, years_loaded, n_pentads_year)
                else:
                    for thr in EVENT_CFG["post20_drop_thresholds"]:
                        add_milestone_cols(rec, f"fd_drop_to_{int(thr)}", None, years_loaded, n_pentads_year)

                if min_index is not None:
                    rec_end_for_milestones = full_recovery40 if full_recovery40 is not None else (n_total - 1)
                    for thr in EVENT_CFG["recovery_milestone_thresholds"]:
                        idx_thr = first_idx_geq(ts, min_index, rec_end_for_milestones, thr)
                        add_milestone_cols(rec, f"fd_rec_to_{int(thr)}", idx_thr, years_loaded, n_pentads_year)
                else:
                    for thr in EVENT_CFG["recovery_milestone_thresholds"]:
                        add_milestone_cols(rec, f"fd_rec_to_{int(thr)}", None, years_loaded, n_pentads_year)

            events.append(rec)

            if full_recovery40 is not None:
                t = max(full_recovery40 + 1 + EVENT_CFG["post_event_skip_pentads"], t + 1)
            else:
                t = max(final_recovery20 + 1 + EVENT_CFG["post_event_skip_pentads"], t + 1)

    print(f"\n[OK] Total flash drought events detected: {len(events):,}")

    if len(events) == 0:
        print("[INFO] No events detected with the current criteria.")
    else:
        df_events = pd.DataFrame(events)

        if EVENT_CFG["attach_lat_lon"] and meta_ref is not None:
            if ("latitude" in meta_ref.files) and ("longitude" in meta_ref.files):
                lats = meta_ref["latitude"]
                lons = meta_ref["longitude"]
                df_events["latitude"] = df_events["point_idx"].map(lambda i: float(lats[int(i)]))
                df_events["longitude"] = df_events["point_idx"].map(lambda i: float(lons[int(i)]))

        out_csv = EVENT_CFG["event_output_dir"] / EVENT_CFG["output_csv_name"]

        if out_csv.exists() and EVENT_CFG["overwrite_output_csv"]:
            ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_csv = EVENT_CFG["event_output_dir"] / f"{EVENT_CFG['backup_prefix']}_{ts_tag}.csv"
            out_csv.replace(backup_csv)
            print(f"[INFO] Existing output backed up to: {backup_csv}")

        df_events.to_csv(out_csv, index=False)
        print(f"[OK] Saved events CSV: {out_csv}")

        print("\n=== EVENT TABLE PREVIEW ===")
        print(df_events.head(10))

def main():
    run_daily_to_pentad()
    run_percentile_stage()
    run_event_stage()


if __name__ == "__main__":
    main()

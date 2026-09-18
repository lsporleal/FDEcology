import pandas as pd

# Load phases
phases_df = pd.read_csv(r"D:\UNL\FD&Ecosystem\DATA\NEW\FD_rzsm_3C_1Dr_I_2000_no_relapse_updated.csv")

date_cols = [
    'onset_start_date', 'onset_end_date',
    'persistence_start_date', 'persistence_end_date',
    'cooldown_start_date', 'cooldown_end_date'
]
for col in date_cols:
    phases_df[col] = pd.to_datetime(phases_df[col])

# Reshape phases into long format ONCE
onset = phases_df[['latitude', 'longitude', 'onset_start_date', 'onset_end_date']].copy()
onset.columns = ['latitude', 'longitude', 'phase_start', 'phase_end']
onset['phase'] = 1

persistence = phases_df[['latitude', 'longitude', 'persistence_start_date', 'persistence_end_date']].copy()
persistence.columns = ['latitude', 'longitude', 'phase_start', 'phase_end']
persistence['phase'] = 2

cooldown = phases_df[['latitude', 'longitude', 'cooldown_start_date', 'cooldown_end_date']].copy()
cooldown.columns = ['latitude', 'longitude', 'phase_start', 'phase_end']
cooldown['phase'] = 3

phases_long = pd.concat([onset, persistence, cooldown], ignore_index=True)

# ── Process NDVI in chunks ──
CHUNK_SIZE = 100_000  # Adjust down to 50_000 if still crashing
ndvi_path = r"D:\UNL\FD&Ecosystem\DATA\NEW\NDVI\NDVI_Time_Linear_Interpolated.csv"
output_path = r"D:\UNL\FD&Ecosystem\DATA\NEW\NDVI\NDVI_Time_Linear_Interpolated_with_phases.csv"
phase_name_map = {0: 'normal', 1: 'onset', 2: 'persistence', 3: 'cooldown'}

first_chunk = True

for i, chunk in enumerate(pd.read_csv(ndvi_path, chunksize=CHUNK_SIZE)):
    print(f"Processing chunk {i+1}...")

    chunk['Date'] = pd.to_datetime(chunk['Date'])

    # Merge chunk with phases
    merged = chunk.merge(phases_long, on=['latitude', 'longitude'], how='left')

    # Filter to rows where date falls in phase window
    in_phase = (merged['Date'] >= merged['phase_start']) & (merged['Date'] <= merged['phase_end'])
    matched = merged[in_phase].drop_duplicates(subset=['Date', 'latitude', 'longitude'])

    # Join phase back to chunk
    chunk = chunk.merge(
        matched[['Date', 'latitude', 'longitude', 'phase']],
        on=['Date', 'latitude', 'longitude'],
        how='left'
    )

    chunk['phase'] = chunk['phase'].fillna(0).astype(int)
    chunk['phase_name'] = chunk['phase'].map(phase_name_map)

    # Write to CSV (append after first chunk)
    chunk.to_csv(output_path, index=False, mode='w' if first_chunk else 'a', header=first_chunk)
    first_chunk = False

print("Done! File saved to:", output_path)
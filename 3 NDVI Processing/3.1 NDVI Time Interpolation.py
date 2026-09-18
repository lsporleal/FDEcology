# -*- coding: utf-8 -*-
"""
Created on Sun Feb 22 19:35:28 2026

TIME INTERPOLATION OF NDVI_MODIS DATA

@author: REVS

"""


import pandas as pd
import numpy as np

# --------------------------------------------------
# 1. Read Files
# --------------------------------------------------
ndvi_df = pd.read_csv(r"D:\UNL\FD&Ecosystem\DATA\NDVI_MODIS_DATA_NEW\MOD13Q1_16d.csv")
pentad_df = pd.read_csv(r"D:\UNL\FD&Ecosystem\DATA\Pentad_Calendar_1980_2024.csv")

# Convert to datetime
ndvi_df['date'] = pd.to_datetime(ndvi_df['date'])
pentad_df['Date'] = pd.to_datetime(pentad_df['Date'])

# --------------------------------------------------
# 2. Filter years (2000–2012)
# --------------------------------------------------
pentad_df = pentad_df[
    (pentad_df['Year'] >= 2000) &
    (pentad_df['Year'] <= 2015)
]

# Sort NDVI properly
ndvi_df = ndvi_df.sort_values(['latitude', 'longitude', 'date'])

# --------------------------------------------------
# 3. Interpolate grid-wise
# --------------------------------------------------
results = []

for (lat, lon), group in ndvi_df.groupby(['latitude', 'longitude']):
    
    group = group.set_index('date').sort_index()
    
    # Combine original NDVI dates + pentad dates
    combined_index = group.index.union(pentad_df['Date'])
    
    group_full = group.reindex(combined_index)
    
    # Time-based linear interpolation
    group_full['NDVI'] = group_full['NDVI'].interpolate(
        method='time',
        limit_direction='both'
    )
    
    # Extract only pentad dates
    interpolated = group_full.loc[pentad_df['Date']].copy()
    
    interpolated['latitude'] = lat
    interpolated['longitude'] = lon
    interpolated['Date'] = interpolated.index
    
    # Merge pentad info
    interpolated = interpolated.merge(
        pentad_df[['Year','Pentad','Date']],
        on='Date',
        how='left'
    )
    
    results.append(
        interpolated[['Year','Pentad','Date','latitude','longitude','NDVI']]
    )

# --------------------------------------------------
# 4. Combine all grids
# --------------------------------------------------
final_df = pd.concat(results, ignore_index=True)

# Save output
final_df.to_csv(r"D:\UNL\FD&Ecosystem\DATA\NDVI_MODIS_DATA_NEW\NDVI_Time_Linear_Interpolated.csv", index=False)

print("Pentad interpolated NDVI file (2000–2012) created successfully.")

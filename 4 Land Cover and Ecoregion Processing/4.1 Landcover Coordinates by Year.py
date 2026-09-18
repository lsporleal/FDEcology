# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 10:27:17 2026

@author: leasp
"""



import pandas as pd
import numpy as np
import glob
import os
from scipy.spatial import cKDTree
from collections import defaultdict

points = pd.read_csv("D:/UNL/CCI LandCoverGrid/FDCoord/unique_lat_lon_points.csv")

point_coords = points[["lat","lon"]].values
coord_strings = points["lat"].astype(str) + "," + points["lon"].astype(str)

files = sorted(glob.glob("D:/UNL/CCI LandCoverGrid/*.csv"))

# -------------------------------------------------
# BUILD KDTree USING ONLY FIRST YEAR


first_file = files[0]

grid_coords = pd.read_csv(first_file, usecols=["lat","lon"]).values

tree = cKDTree(grid_coords)

dist, idx = tree.query(point_coords)



# -------------------------------------------------
# LOOP YEARS USING SAME INDEX


results = {}

for f in files:

    year = os.path.basename(f).replace(".csv","")

    # load only land cover column
    nlcd = pd.read_csv(f, usecols=["lccs_class"])

    classes = nlcd.iloc[idx]["lccs_class"].values

    class_dict = defaultdict(list)

    for lc, coord in zip(classes, coord_strings):
        class_dict[lc].append(coord)

    results[year] = class_dict

# -------------------------------------------------
# BUILD TABLE


all_classes = sorted({c for y in results for c in results[y]})

df = pd.DataFrame(index=all_classes)

for year in results:
    df[year] = [
        "; ".join(results[year].get(cls, []))
        for cls in all_classes
    ]

df.index.name = "lccs_class"
df.columns = [ '2016', '2017', '2018', '2019',
                  '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007',
                  '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']

df.to_csv("D:/UNL/CCI LandCoverGrid/FDCoord/landcover_coordinates_by_year.csv")

dcheck = pd.read_csv("D:/UNL/CCI LandCoverGrid/FDCoord/landcover_coordinates_by_year.csv")
dcheck.columns = ['lccs_class', '2016', '2017', '2018', '2019',
                  '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007',
                  '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']

dcheck.to_csv("D:/UNL/CCI LandCoverGrid/FDCoord/landcover_coordinates_by_year.csv")

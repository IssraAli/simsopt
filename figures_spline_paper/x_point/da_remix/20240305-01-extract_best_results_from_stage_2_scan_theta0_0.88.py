#!/usr/bin/env python

import glob
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from paretoset import paretoset

# Initialize an empty DataFrame
df = pd.DataFrame()

results = glob.glob("stage_2_scan_twin_out/order*/results.json")
#df = None
for results_file in results:
    with open(results_file, "r") as f:
        data = json.load(f)

    # Wrap lists in another list
    for key, value in data.items():
        if isinstance(value, list):
            data[key] = [value]

    df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)

print(df)
print("keys:", df.keys())
print("nfp:", df.nfp)
print("dir:", dir(df))
print(df["Jf"])
#print(df["linking_number"])
print(df.linking_number)
succeeded = df["linking_number"] < 0.1
# succeeded = np.logical_and(succeeded, df["Jf"] < 5e-3)
# succeeded = np.logical_and(succeeded, df["max_max_curvature"] < 12)
succeeded = np.logical_and(succeeded, df["coil_coil_distance"] > 0.03)
# np.logical_and(
#     df["linking_number"] < 0.1, 
#     df["Jf"] < 5e-3),
# df["max_max_curvature"] < 12)

print(succeeded)
df_filtered = df[succeeded]
print("After masking:")

print(df_filtered["linking_number"])
succeeded = df_filtered["linking_number"] < 0.1
print(succeeded)

index = np.argmin(df_filtered["Jf"])
print("index with lowest B dot n error:", index)
#print("lowest B dot n error:", df_filtered["Jf"][index])

pareto_mask = paretoset(df_filtered[["Jf", "max_max_curvature"]],
                sense=[min, min])
print("Pareto mask:")
print(pareto_mask)
df_pareto = df_filtered[pareto_mask]

print("Best Pareto-optimal results:")
print(df_pareto[["directory", "Jf", "max_max_curvature", "length", "max_mean_squared_curvature", "coil_coil_distance"]])
print("Directory names only:")
for dirname in df_pareto["directory"]:
    print(dirname)

plt.figure(figsize=(14.5, 8))
plt.rc('font', size=8)
nrows = 3
ncols = 5
markersize = 5

subplot_index = 1
plt.subplot(nrows, ncols, subplot_index)
subplot_index += 1
plt.scatter(df["Jf"], df["max_max_curvature"], c=df["length"], s=1)
plt.colorbar(label="length")
plt.scatter(df_filtered["Jf"], df_filtered["max_max_curvature"], c=df_filtered["length"], s=markersize)
plt.scatter(df_pareto["Jf"], df_pareto["max_max_curvature"], c=df_pareto["length"], marker="+")
plt.xlabel('Bnormal objective')
plt.ylabel('Max curvature')
plt.xscale("log")

plt.subplot(nrows, ncols, subplot_index)
subplot_index += 1
plt.scatter(df_filtered["length_target"], df_filtered["length"], c=df_filtered["Jf"], s=markersize, norm=colors.LogNorm())
plt.colorbar(label="Bnormal objective")
plt.xlabel('length_target')
plt.ylabel('length')

plt.subplot(nrows, ncols, subplot_index)
subplot_index += 1
plt.scatter(df_filtered["max_curvature_threshold"], df_filtered["max_max_curvature"], c=df_filtered["Jf"], s=markersize, norm=colors.LogNorm())
plt.colorbar(label="Bnormal objective")
plt.xlabel('max_curvature_threshold')
plt.ylabel('max_max_curvature')

def plot_2d_hist(field, log=False):
    global subplot_index
    plt.subplot(nrows, ncols, subplot_index)
    subplot_index += 1
    nbins = 20
    if log:
        data = df[field]
        bins = np.logspace(np.log10(data.min()), np.log10(data.max()), nbins)
    else:
        bins = nbins
    plt.hist(df[field], bins=bins, label="before filtering")
    plt.hist(df_filtered[field], bins=bins, alpha=1, label="after filtering")
    plt.xlabel(field)
    plt.legend(loc=0, fontsize=6)
    if log:
        plt.xscale("log")

fields = (
    ("length", False),
    ("length_target", False),
    ("max_curvature_threshold", False),
    ("max_curvature_weight", True),
    ("max_max_curvature", False),
    ("coil_coil_distance", False),
    ("order", False),
    ("R1", False),
    ("msc_threshold", False),
    ("msc_weight", True),
)

for field, log in fields:
    plot_2d_hist(field, log)

"""
plt.subplot(nrows, ncols, 2)
nbins = 20
plt.hist(df["length"], bins=nbins, label="before filtering")
plt.hist(df_filtered["length"], bins=nbins, alpha=0.5, label="after filtering")
plt.legend(loc=0, fontsize=7)
plt.xlabel('length')

plt.subplot(nrows, ncols, 3)
nbins = 20
plt.hist(df["max_max_curvature"], bins=nbins, label="before filtering")
plt.hist(df_filtered["max_max_curvature"], bins=nbins, alpha=0.5, label="after filtering")
plt.legend(loc=0, fontsize=7)
plt.xlabel('Max curvature')
"""

plt.figtext(0.5, 0.995, os.path.abspath(__file__), ha="center", va="top", fontsize=6)
plt.tight_layout()
plt.show()

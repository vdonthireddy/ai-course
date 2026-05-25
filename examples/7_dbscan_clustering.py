#!/usr/bin/env python3
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

print("====================================================")
# 1. Prepare data (A dense group of points + a separate outlier/noise point)
# Coordinates in 2D space
X = np.array([
    [1.0, 1.0], [1.1, 0.9], [0.9, 1.1], [1.0, 1.2], # Dense Cluster A
    [5.0, 5.0], [5.1, 4.9], [4.9, 5.1],             # Dense Cluster B
    [10.0, 10.0]                                    # Outlier / Noise Point
])

print("DBSCAN Coordinate Dataset:")
for idx, pt in enumerate(X):
    print(f" - Point {idx+1}: ({pt[0]}, {pt[1]})")

# 2. Fit DBSCAN
# eps = 0.5 (maximum search distance between neighbors)
# min_samples = 3 (minimum points in neighborhood to form a cluster)
dbscan = DBSCAN(eps=0.5, min_samples=3)
labels = dbscan.fit_predict(X)

# 3. Output results
print("\nDBSCAN Results (Note: label -1 represents Outlier/Noise):")
for idx, label in enumerate(labels):
    pt_type = "Noise (Outlier)"
    if label != -1:
        # Check if point is core (its index is in core_sample_indices_)
        if idx in dbscan.core_sample_indices_:
            pt_type = "Core Point"
        else:
            pt_type = "Border Point"
    print(f" - Point {idx+1} ({X[idx][0]}, {X[idx][1]}): Label = {label} | Classification: {pt_type}")

# 4. Generate and save visualization plot
os.makedirs("plots", exist_ok=True)
plt.figure(figsize=(9, 6))

core_indices = set(dbscan.core_sample_indices_)
colors = ["#4F46E5", "#10B981"]
legend_labels = {}

for idx, label in enumerate(labels):
    if label == -1:
        lbl = "Noise (Outlier)"
        h = plt.scatter(X[idx, 0], X[idx, 1], color="#EF4444", s=150, marker="x", linewidths=2.5, zorder=4)
        legend_labels[lbl] = h
    else:
        pt_color = colors[label % len(colors)]
        if idx in core_indices:
            lbl = f"Cluster {label} (Core Point)"
            h = plt.scatter(X[idx, 0], X[idx, 1], color=pt_color, s=150, marker="o", edgecolor="black", linewidths=1.5, zorder=3)
            legend_labels[lbl] = h
        else:
            lbl = f"Cluster {label} (Border Point)"
            h = plt.scatter(X[idx, 0], X[idx, 1], color=pt_color, s=100, marker="^", edgecolor="black", linewidths=1.5, zorder=3)
            legend_labels[lbl] = h

# Set grid, labels, title and unique legend
plt.title("DBSCAN Spatial Density Clustering (eps=0.5, min_samples=3)", fontsize=13, fontweight="bold", pad=15)
plt.xlabel("X Coordinate Space", fontsize=11, labelpad=10)
plt.ylabel("Y Coordinate Space", fontsize=11, labelpad=10)
plt.grid(True, linestyle="--", alpha=0.5)

sorted_keys = sorted(legend_labels.keys())
plt.legend([legend_labels[k] for k in sorted_keys], sorted_keys, loc="center left", frameon=True, facecolor="white", edgecolor="#E5E7EB")
plt.tight_layout()

# Save the plot
plot_path = "plots/examples_7_dbscan.png"
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"\nVisual plot successfully saved to: {plot_path}")
print("====================================================")

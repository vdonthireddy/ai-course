#!/usr/bin/env python3
import numpy as np
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
print("====================================================")

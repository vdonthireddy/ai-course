#!/usr/bin/env python3
import math
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

"""
DBSCAN Density Clustering from Scratch (No Frameworks)
======================================================

Example Used:
-------------
We want to cluster 2D coordinate points and identify noise (outliers) based on density.
Our dataset consists of 8 points:
  - Cluster A (dense): [1.0, 1.0], [1.1, 0.9], [0.9, 1.1], [1.0, 1.2]
  - Cluster B (dense): [5.0, 5.0], [5.1, 4.9], [4.9, 5.1]
  - Outlier (noise):   [10.0, 10.0]

DBSCAN Parameters:
------------------
* **eps** (Epsilon radius) = 0.5: Maximum distance to search for neighboring points.
* **min_samples** = 3: Minimum points required in epsilon neighborhood to form a Core Point.

Algorithmic Steps:
------------------
1. **Find Neighbors**: For each point, find all other points within distance `eps`.
2. **Identify Core Points**: If neighbors count >= `min_samples`, classify as a Core Point.
3. **Expand Clusters**: For each unvisited core point, start a new cluster and recursively 
   add all reachable points in its neighborhood.
4. **Label Noise**: Points that are never reached by any cluster split are labeled as Outliers/Noise (-1).
"""

# 1. Dataset
X = [
    [1.0, 1.0], [1.1, 0.9], [0.9, 1.1], [1.0, 1.2],  # Dense Group 1
    [5.0, 5.0], [5.1, 4.9], [4.9, 5.1],              # Dense Group 2
    [10.0, 10.0]                                     # Outlier Point
]
n_points = len(X)

# Parameters
eps = 0.5
min_samples = 3

print("====================================================")
print("DBSCAN Dataset Coordinates:")
for idx, pt in zip(range(n_points), X):
    print(f" - Point {idx+1}: ({pt[0]}, {pt[1]})")

# Helper: Euclidean Distance
def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Step A: Find all neighbors for each point
neighbors_list = []
for i in range(n_points):
    neighbors = []
    for j in range(n_points):
        if get_distance(X[i], X[j]) <= eps:
            neighbors.append(j)
    neighbors_list.append(neighbors)

# Step B: Identify Core Points
is_core = [False] * n_points
for i in range(n_points):
    # If neighborhood size is >= min_samples, it is a core point
    if len(neighbors_list[i]) >= min_samples:
        is_core[i] = True

print("\nPoint Classifications (Core vs Border/Noise check):")
for i in range(n_points):
    class_str = "Core Point" if is_core[i] else "Border / Noise candidate"
    print(f" - Point {i+1} has {len(neighbors_list[i])} neighbors -> {class_str}")

# Step C: Cluster expansion
labels = [-1] * n_points  # Initial labels. -1 represents Noise (outliers)
visited = [False] * n_points
current_cluster_id = 0

for i in range(n_points):
    if visited[i]:
        continue
    
    # We only initiate cluster growth from Core Points
    if is_core[i]:
        # Start a new cluster
        labels[i] = current_cluster_id
        visited[i] = True
        
        # Queue for cluster search expansion (Breadth-First Search)
        queue = list(neighbors_list[i])
        
        # Process the queue
        idx = 0
        while idx < len(queue):
            neighbor_idx = queue[idx]
            
            if not visited[neighbor_idx]:
                visited[neighbor_idx] = True
                labels[neighbor_idx] = current_cluster_id
                
                # If neighbor is also a core point, add its neighbors to search queue
                if is_core[neighbor_idx]:
                    for item in neighbors_list[neighbor_idx]:
                        if item not in queue:
                            queue.append(item)
            elif labels[neighbor_idx] == -1:
                # If point was labeled as noise but is reachable, assign to this cluster (Border point!)
                labels[neighbor_idx] = current_cluster_id
                
            idx += 1
            
        current_cluster_id += 1 # Increment cluster ID for next group

# 4. Print final results
print("\nFinal DBSCAN Clustering Outputs:")
for i in range(n_points):
    status = f"Cluster {labels[i]}" if labels[i] != -1 else "Noise (Outlier)"
    core_lbl = "Core" if is_core[i] else ("Border" if labels[i] != -1 else "Noise")
    print(f" - Point {i+1:2d} ({X[i][0]:4.1f}, {X[i][1]:4.1f}) -> Assigned to: {status:12s} | Type: {core_lbl}")

# 5. Generate and save visualization plot
os.makedirs("plots", exist_ok=True)
plt.figure(figsize=(9, 6))

colors = ["#4F46E5", "#10B981"]
legend_labels = {}

for idx in range(n_points):
    label = labels[idx]
    pt = X[idx]
    if label == -1:
        lbl = "Noise (Outlier)"
        h = plt.scatter(pt[0], pt[1], color="#EF4444", s=150, marker="x", linewidths=2.5, zorder=4)
        legend_labels[lbl] = h
    else:
        pt_color = colors[label % len(colors)]
        if is_core[idx]:
            lbl = f"Cluster {label} (Core Point)"
            h = plt.scatter(pt[0], pt[1], color=pt_color, s=150, marker="o", edgecolor="black", linewidths=1.5, zorder=3)
            legend_labels[lbl] = h
        else:
            lbl = f"Cluster {label} (Border Point)"
            h = plt.scatter(pt[0], pt[1], color=pt_color, s=100, marker="^", edgecolor="black", linewidths=1.5, zorder=3)
            legend_labels[lbl] = h

plt.title("DBSCAN Density Clustering from Scratch (eps=0.5, min_samples=3)", fontsize=13, fontweight="bold", pad=15)
plt.xlabel("X Coordinate", fontsize=11, labelpad=10)
plt.ylabel("Y Coordinate", fontsize=11, labelpad=10)
plt.grid(True, linestyle="--", alpha=0.5)

sorted_keys = sorted(legend_labels.keys())
plt.legend([legend_labels[k] for k in sorted_keys], sorted_keys, loc="center left", frameon=True, facecolor="white", edgecolor="#E5E7EB")
plt.tight_layout()

# Save the plot
plot_path = "plots/scratch_6_dbscan.png"
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"\nVisual plot successfully saved to: {plot_path}")
print("====================================================")

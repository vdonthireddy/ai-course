#!/usr/bin/env python3
import math

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

# Output ASCII spatial density coordinate space plot
print("\nVisual Density Space Layout:")
print("  Y-coordinate")
print("  ^")
print(" 10|                                  Noise (Point 8: 10, 10)")
print("   |")
print("  5|          Cluster 1 (Points 5-7: ~5, 5)")
print("   |")
print("  1|  Cluster 0 (Points 1-4: ~1, 1)")
print("  0+-----------------------------------> X-coordinate")
print("     1   2   3   4   5   6   7   8   9   10")
print("====================================================")

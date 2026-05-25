#!/usr/bin/env python3
import math

"""
K-Means Clustering from Scratch (No Frameworks)
==============================================

Example Used:
-------------
We want to group 10 customers into 3 clusters (K = 3) based on two features:
  - Annual Income (in $k)
  - Spending Score (1 to 100)

Dataset of 10 customers:
  - Cust 1:  Income = 15, Spending = 39
  - Cust 2:  Income = 16, Spending = 81
  - Cust 3:  Income = 17, Spending = 6
  - Cust 4:  Income = 18, Spending = 77
  - Cust 5:  Income = 19, Spending = 40
  - Cust 6:  Income = 70, Spending = 50
  - Cust 7:  Income = 72, Spending = 48
  - Cust 8:  Income = 75, Spending = 52
  - Cust 9:  Income = 90, Spending = 97
  - Cust 10: Income = 95, Spending = 92

Mathematical Goal:
------------------
1. **Initialize Centroids**: Place 3 points (centroids) in the data space.
2. **Assign Clusters**: Assign each customer to the closest centroid (using Euclidean distance).
3. **Update Centroids**: Calculate the arithmetic mean of all points assigned to each cluster and move the centroid there.
4. **Iterate**: Repeat steps 2 and 3 until centroids stop moving (convergence).
"""

# 1. Dataset (Annual Income, Spending Score)
X = [
    [15, 39], [16, 81], [17, 6],  [18, 77], [19, 40],
    [70, 50], [72, 48], [75, 52], [90, 97], [95, 92]
]
n_points = len(X)
K = 3

print("====================================================")
print("Customer Dataset:")
for idx, pt in enumerate(X):
    print(f" - Customer {idx+1:2d}: Income ${pt[0]}k, Spending Score: {pt[1]}")

# Helper: Euclidean distance
def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# 2. Initialize Centroids (using fixed points from the dataset for reproducibility)
# In practice, these can be random or chosen via K-Means++
centroids = [
    [15, 39],  # Centroid 0 (takes Cust 1 initial coordinates)
    [70, 50],  # Centroid 1 (takes Cust 6 initial coordinates)
    [90, 97]   # Centroid 2 (takes Cust 9 initial coordinates)
]

print("\nInitial Centroids:")
for i, c in enumerate(centroids):
    print(f" - Centroid {i}: ({c[0]}, {c[1]})")

# 3. K-Means Main Loop
epochs = 10
for epoch in range(epochs):
    # Cluster assignments list (stores centroid index for each data point)
    assignments = [-1] * n_points
    
    # --- Step A: Assign points to nearest centroid ---
    for i in range(n_points):
        min_dist = 999999.0
        best_cluster = -1
        for c_idx in range(K):
            dist = euclidean_distance(X[i], centroids[c_idx])
            if dist < min_dist:
                min_dist = dist
                best_cluster = c_idx
        assignments[i] = best_cluster
        
    # --- Step B: Update Centroids to be the mean of assigned points ---
    new_centroids = [[0.0, 0.0] for _ in range(K)]
    counts = [0] * K
    
    # Accumulate sums
    for i in range(n_points):
        c_idx = assignments[i]
        new_centroids[c_idx][0] += X[i][0]
        new_centroids[c_idx][1] += X[i][1]
        counts[c_idx] += 1
        
    # Divide to compute averages
    changed = False
    for c_idx in range(K):
        if counts[c_idx] > 0:
            mean_x = new_centroids[c_idx][0] / counts[c_idx]
            mean_y = new_centroids[c_idx][1] / counts[c_idx]
            
            # Check if centroid has shifted
            if mean_x != centroids[c_idx][0] or mean_y != centroids[c_idx][1]:
                changed = True
                
            centroids[c_idx] = [mean_x, mean_y]
            
    print(f" - Iteration {epoch+1:2d} completed. Centroids shifted? {changed}")
    if not changed:
        print(f" -> Centroids converged early at iteration {epoch+1}!")
        break

# 4. Output final clusters
print("\nFinal Optimized Centroids (Segment Profiles):")
for i, c in enumerate(centroids):
    print(f" - Segment {i} Centroid: Income ${c[0]:.2f}k, Spending Score: {c[1]:.2f}")

print("\nFinal Customer Groupings:")
for i in range(n_points):
    print(f" - Customer {i+1:2d} ({X[i][0]}, {X[i][1]}) -> Assigned to Segment {assignments[i]}")

# 5. Predict segment for a new customer
new_customer = [80, 55]
best_cluster = -1
min_dist = 999999.0
for c_idx in range(K):
    dist = euclidean_distance(new_customer, centroids[c_idx])
    if dist < min_dist:
        min_dist = dist
        best_cluster = c_idx

print(f"\nNew Customer Profile (Income: $80k, Spending: 55):")
print(f" - Predicted Segment Classification: Segment {best_cluster}")

# Output ASCII cluster segments plot
print("\nVisual Plot of Clusters (Income vs Spending Score):")
print("Spending Score")
print(" ^")
print("90|          [Segment 2] (High Income, High Spending)")
print("60|                   [Segment 1] (Mid Income, Mid Spending) <-- New Customer Segment")
print("30|   [Segment 0] (Low Income, Low/High Spending)")
print("  +---------------------------------------------> Annual Income")
print("     $15k     $45k     $75k     $105k")
print("====================================================")

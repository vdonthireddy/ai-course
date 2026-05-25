#!/usr/bin/env python3
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

print("====================================================")
# 1. Prepare data (Features: [Annual Income in $k, Spending Score 1-100])
X = np.array([
    [15, 39], [16, 81], [17, 6],  [18, 77], [19, 40],
    [70, 50], [72, 48], [75, 52], [78, 45], [80, 50],
    [90, 97], [95, 92], [100, 95], [105, 90], [110, 98]
])

print("Customer Segmentation Dataset:")
for idx, pt in enumerate(X):
    print(f" - Cust {idx+1}: Income ${pt[0]}k, Spending Score: {pt[1]}")

# 2. Fit K-Means Clustering with K=3
# We initialize 3 clusters representing:
#  - Low Income, Low/High Spending (bottom-left / top-left)
#  - Mid Income, Mid Spending (middle)
#  - High Income, High Spending (top-right)
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
labels = kmeans.fit_predict(X)

# 3. Output results
print("\nK-Means Cluster Centroids (Representing Segment Profiles):")
centroids = kmeans.cluster_centers_
for idx, cent in enumerate(centroids):
    print(f" - Segment {idx} Centroid: Income ${cent[0]:.2f}k, Spending Score: {cent[1]:.2f}")

print("\nCustomer Assignments:")
for idx, label in enumerate(labels):
    print(f" - Customer {idx+1} is assigned to Segment {label}")

# 4. Predict segment for a new customer: Income 85k, Spending Score 60
new_cust = np.array([[85, 60]])
pred_segment = kmeans.predict(new_cust)[0]
print(f"\nNew Customer (Income: $85k, Spending: 60) Predicted Segment: {pred_segment}")

# 5. Generate and save visualization plot
os.makedirs("plots", exist_ok=True)
plt.figure(figsize=(9, 6))

colors = ["#4F46E5", "#10B981", "#EC4899"]
for k in range(3):
    plt.scatter(X[labels == k, 0], X[labels == k, 1], color=colors[k], s=100, marker="o", edgecolor="black", label=f"Segment {k}", zorder=3)

# Plot centroids
plt.scatter(centroids[:, 0], centroids[:, 1], color="#F59E0B", s=250, marker="X", edgecolor="black", linewidths=1.5, label="Centroids", zorder=4)

# Plot new customer query point
plt.scatter(new_cust[0, 0], new_cust[0, 1], color="#EF4444", s=250, marker="*", edgecolor="black", label=f"New Customer (Segment {pred_segment})", zorder=5)

plt.title("K-Means Customer Segmentation (K=3)", fontsize=13, fontweight="bold", pad=15)
plt.xlabel("Annual Income ($k)", fontsize=11, labelpad=10)
plt.ylabel("Spending Score (1-100)", fontsize=11, labelpad=10)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#E5E7EB")
plt.tight_layout()

# Save the plot
plot_path = "plots/examples_6_kmeans.png"
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"\nVisual plot successfully saved to: {plot_path}")
print("====================================================")

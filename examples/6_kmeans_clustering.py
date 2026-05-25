#!/usr/bin/env python3
import numpy as np
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
print("====================================================")

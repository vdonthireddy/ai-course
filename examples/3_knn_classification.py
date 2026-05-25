#!/usr/bin/env python3
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

print("====================================================")
# 1. Prepare data (Features: [Age, Income in $k], Target: Accepted offer 1=Yes, 0=No)
X = np.array([
    [22, 35], [25, 40], [45, 120], [50, 150], [30, 60],
    [32, 55], [40, 90],  [28, 48],  [55, 130], [21, 30]
])
y = np.array([0, 0, 1, 1, 0, 0, 1, 0, 1, 0])

print("Credit Card Offer Acceptance Dataset (Age & Income):")
for feat, target in zip(X, y):
    status = "Accepted" if target == 1 else "Declined"
    print(f" - Age: {feat[0]}, Income: ${feat[1]}k -> {status}")

# 2. Scale features (KNN is highly sensitive to feature scales!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Fit KNN Classifier with K=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_scaled, y)

# 4. Predict offering outcome for a new customer: Age 35, Income 95k
new_customer = np.array([[35, 95]])
new_customer_scaled = scaler.transform(new_customer)

pred_class = knn.predict(new_customer_scaled)[0]
outcome = "Accepted" if pred_class == 1 else "Declined"

# 5. Output result and neighbors
distances, indices = knn.kneighbors(new_customer_scaled)
print(f"\nPrediction for new customer (Age: 35, Income: $95k):")
print(f" - Predicted Offer Decision: {outcome}")
print("\nNearest Neighbors details:")
for dist, idx in zip(distances[0], indices[0]):
    neighbor_status = "Accepted" if y[idx] == 1 else "Declined"
    print(f" - Neighbor (Index {idx}): Age {X[idx][0]}, Income ${X[idx][1]}k | Status: {neighbor_status} (Scaled Distance: {dist:.4f})")
print("====================================================")

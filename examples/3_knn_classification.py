#!/usr/bin/env python3
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

# 6. Generate and save visualization plot
os.makedirs("plots", exist_ok=True)
plt.figure(figsize=(9, 6))

# Plot training data points in standardized space
plt.scatter(X_scaled[y == 0, 0], X_scaled[y == 0, 1], color="#EF4444", s=100, marker="o", edgecolor="black", label="Declined Offer (0)", zorder=3)
plt.scatter(X_scaled[y == 1, 0], X_scaled[y == 1, 1], color="#10B981", s=100, marker="o", edgecolor="black", label="Accepted Offer (1)", zorder=3)

# Plot new customer query point
plt.scatter(new_customer_scaled[0, 0], new_customer_scaled[0, 1], color="#F59E0B", s=250, marker="*", edgecolor="black", label="New Customer (Age 35, $95k)", zorder=5)

# Connect query point to its nearest neighbors in scaled space
for idx in indices[0]:
    neighbor_scaled = X_scaled[idx]
    plt.plot([new_customer_scaled[0, 0], neighbor_scaled[0]], [new_customer_scaled[0, 1], neighbor_scaled[1]], color="#6B7280", linestyle=":", linewidth=1.5, zorder=2)
    # Circle the neighbor to highlight it
    plt.scatter(neighbor_scaled[0], neighbor_scaled[1], s=200, facecolors='none', edgecolors='#4F46E5', linewidths=2, zorder=4)

plt.title("KNN Classification (Standardized Age & Income Feature Space, K=3)", fontsize=13, fontweight="bold", pad=15)
plt.xlabel("Age (Standardized)", fontsize=11, labelpad=10)
plt.ylabel("Income (Standardized)", fontsize=11, labelpad=10)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#E5E7EB")
plt.tight_layout()

# Save the plot
plot_path = "plots/examples_3_knn.png"
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"\nVisual plot successfully saved to: {plot_path}")
print("====================================================")

#!/usr/bin/env python3
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier

print("====================================================")
# 1. Prepare data (Features: [Age, Tenure in months, Calls to Support])
# Target: Churn status (1 = Churned, 0 = Remained loyal)
X = np.array([
    [25, 2, 8],  # Young, short tenure, high support calls -> Churn
    [45, 36, 1], # Older, long tenure, low support calls -> Loyal
    [30, 12, 4], # Average
    [22, 1, 9],  # Churn
    [55, 48, 0], # Loyal
    [35, 24, 2], # Loyal
    [28, 6, 6],  # Churn
    [40, 18, 3], # Loyal
    [50, 40, 1], # Loyal
    [26, 3, 7]   # Churn
])
y = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1])

# 2. Fit Decision Tree Classifier (simple tree)
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X, y)

# 3. Fit Random Forest Classifier (ensemble)
rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(X, y)

# 4. Print Decision Tree rules
print("Trained Decision Tree Rules Structure:")
features = ["Age", "Tenure", "Support_Calls"]
tree_rules = export_text(dt, feature_names=features)
print(tree_rules)

# 5. Check prediction for a risky customer: Age 24, 2 months tenure, 6 calls
risky_customer = np.array([[24, 2, 6]])
dt_pred = dt.predict(risky_customer)[0]
rf_pred = rf.predict(risky_customer)[0]

dt_status = "Churn" if dt_pred == 1 else "Loyal"
rf_status = "Churn" if rf_pred == 1 else "Loyal"

print(f"Prediction for customer (Age: 24, Tenure: 2m, Support Calls: 6):")
print(f" - Decision Tree Prediction : {dt_status}")
print(f" - Random Forest Prediction : {rf_status}")

# 6. Feature importances in Random Forest
print("\nRandom Forest Feature Importances:")
importances = rf.feature_importances_
for feat, imp in zip(features, importances):
    print(f" - Feature '{feat}' Importance: {imp * 100:.2f}%")

# 7. Generate and save visualization plot
os.makedirs("plots", exist_ok=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: Decision Tree Structure Visual Plot
plot_tree(dt, feature_names=features, class_names=["Loyal", "Churn"], filled=True, rounded=True, ax=ax1, fontsize=9)
ax1.set_title("Decision Tree Split Structure (max_depth=3)", fontsize=12, fontweight="bold")

# Subplot 2: Random Forest Feature Importances Horizontal Bar Chart
y_pos = np.arange(len(features))
sorted_indices = np.argsort(importances)
ax2.barh(y_pos, importances[sorted_indices] * 100, color="#4F46E5", edgecolor="#3730A3", height=0.5)
ax2.set_yticks(y_pos)
ax2.set_yticklabels([features[i] for i in sorted_indices], fontsize=10)
ax2.set_xlabel("Relative Importance (%)", fontsize=11, labelpad=10)
ax2.set_title("Random Forest Feature Importances", fontsize=12, fontweight="bold")
ax2.grid(True, axis="x", linestyle="--", alpha=0.5)

plt.suptitle("Tree-Based Models: Decision Tree Structure vs. Random Forest Feature Importances", fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout()

# Save the plot
plot_path = "plots/examples_4_decision_tree.png"
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"\nVisual plot successfully saved to: {plot_path}")
print("====================================================")

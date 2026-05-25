#!/usr/bin/env python3
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

print("====================================================")
# 1. Prepare data (Features: [Credit Score, Debt-to-Income %, Active Loans])
# Target: Default status (1 = Defaulted, 0 = Paid)
X = np.array([
    [750, 0.15, 1],
    [580, 0.45, 4],
    [620, 0.35, 3],
    [800, 0.10, 0],
    [500, 0.60, 5],
    [710, 0.20, 2],
    [690, 0.25, 1],
    [590, 0.50, 4],
    [640, 0.30, 2],
    [780, 0.12, 1]
] * 10) # Multiply to make 100 samples for training
y = np.array([0, 1, 1, 0, 1, 0, 0, 1, 0, 0] * 10)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Fit XGBoost Classifier
xgb_clf = xgb.XGBClassifier(
    n_estimators=30,
    learning_rate=0.1,
    max_depth=3,
    eval_metric="logloss",
    random_state=42
)
xgb_clf.fit(X_train, y_train)

# 3. Predict and evaluate
y_pred = xgb_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"XGBoost Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Paid", "Defaulted"]))

# 4. Predict risk for new borrower: Credit Score 600, Debt-to-Income 40%, 3 active loans
new_borrower = np.array([[600, 0.40, 3]])
pred_prob = xgb_clf.predict_proba(new_borrower)[0][1] # Probability of default (class 1)
print(f"Default Risk for Borrower (Score: 600, DTI: 40%, Loans: 3): {pred_prob * 100:.2f}%")

# 5. Generate and save visualization plot
os.makedirs("plots", exist_ok=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: Feature Importances
features = ["Credit_Score", "Debt_to_Income", "Active_Loans"]
importances = xgb_clf.feature_importances_
y_pos = np.arange(len(features))
sorted_indices = np.argsort(importances)

ax1.barh(y_pos, importances[sorted_indices] * 100, color="#10B981", edgecolor="#047857", height=0.5)
ax1.set_yticks(y_pos)
ax1.set_yticklabels([features[i] for i in sorted_indices], fontsize=10)
ax1.set_xlabel("Relative Importance (%)", fontsize=11, labelpad=10)
ax1.set_title("XGBoost Feature Importances", fontsize=12, fontweight="bold")
ax1.grid(True, axis="x", linestyle="--", alpha=0.5)

# Subplot 2: Risk Probability Curve vs. Credit Score (varying Credit Score, DTI=40%, Loans=3)
credit_scores = np.linspace(400, 850, 200)
synthetic_borrowers = np.zeros((200, 3))
synthetic_borrowers[:, 0] = credit_scores
synthetic_borrowers[:, 1] = 0.40  # Constant DTI
synthetic_borrowers[:, 2] = 3     # Constant Active Loans

default_probs = xgb_clf.predict_proba(synthetic_borrowers)[:, 1]

ax2.plot(credit_scores, default_probs * 100, color="#EF4444", linewidth=3, label="Default Risk Probability")
# Highlight our prediction point (600, 40%, 3)
ax2.scatter([600], [pred_prob * 100], color="#F59E0B", marker="*", s=250, edgecolor="black", label="Borrower (Score 600)", zorder=5)

ax2.set_xlabel("Credit Score", fontsize=11, labelpad=10)
ax2.set_ylabel("Predicted Default Risk (%)", fontsize=11, labelpad=10)
ax2.set_title("Default Risk vs. Credit Score (DTI=40%, Loans=3)", fontsize=12, fontweight="bold")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#E5E7EB")

plt.suptitle("XGBoost Ensemble: Feature Importances & Credit Score Risk Curve", fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout()

# Save the plot
plot_path = "plots/examples_5_xgboost.png"
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"\nVisual plot successfully saved to: {plot_path}")
print("====================================================")

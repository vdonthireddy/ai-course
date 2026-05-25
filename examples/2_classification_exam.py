#!/usr/bin/env python3
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

print("====================================================")
# 1. Prepare data (Hours studied vs Pass/Fail result)
# Features: Hours studied (1D array reshaped to column vector)
# Target: 0 = Fail, 1 = Pass
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

print("Student Exam Dataset (Features = Hours Studied, Target = Pass/Fail):")
for hours, result in zip(X.flatten(), y):
    status = "Pass" if result == 1 else "Fail"
    print(f" - Studied: {hours} hours -> Outcome: {status}")

# 2. Fit Logistic Regression model
clf = LogisticRegression()
clf.fit(X, y)

# 3. Predict outcome and probability for a student who studied 5.5 hours
test_hours = np.array([[5.5]])
pred_class = clf.predict(test_hours)[0]
pred_prob = clf.predict_proba(test_hours)[0] # Returns [prob_fail, prob_pass]

status_pred = "Pass" if pred_class == 1 else "Fail"
print(f"\nPrediction for a student who studied 5.5 hours:")
print(f" - Predicted Class: {status_pred}")
print(f" - Probability of Failing (0): {pred_prob[0]*100:.2f}%")
print(f" - Probability of Passing (1): {pred_prob[1]*100:.2f}%")

# 4. Predict outcome for 3 hours and 8 hours to show difference
print(f"\nQuick Check predictions:")
for h in [3.0, 8.0]:
    p_class = clf.predict([[h]])[0]
    p_prob = clf.predict_proba([[h]])[0][1]
    res = "Pass" if p_class == 1 else "Fail"
    print(f" - Studied {h} hours: Predicted {res} (Confidence: {p_prob*100:.1f}%)")

# 5. Generate and save visualization plot
os.makedirs("plots", exist_ok=True)
plt.figure(figsize=(9, 6))

# Plot training data points
plt.scatter(X[y == 0], y[y == 0], color="#EF4444", s=80, marker="o", edgecolor="black", label="Actual Fail (0)", zorder=3)
plt.scatter(X[y == 1], y[y == 1], color="#10B981", s=80, marker="o", edgecolor="black", label="Actual Pass (1)", zorder=3)

# Plot sigmoid probability curve
X_range = np.linspace(0.5, 10.5, 200).reshape(-1, 1)
y_prob = clf.predict_proba(X_range)[:, 1]
plt.plot(X_range, y_prob, color="#4F46E5", linewidth=3, label="Logistic Probability Curve", zorder=2)

# Highlight prediction for 5.5 hours
plt.scatter([5.5], [pred_prob[1]], color="#F59E0B", marker="*", s=250, edgecolor="black", label="Prediction at 5.5 hours", zorder=5)

# Draw decision boundary (50% probability threshold)
w = clf.coef_[0][0]
b = clf.intercept_[0]
db_hours = -b / w
plt.axvline(x=db_hours, color="#6B7280", linestyle="--", alpha=0.8, label=f"Decision Boundary ({db_hours:.2f} hrs)")

plt.title("Logistic Regression: Exam Pass Probability vs. Study Hours", fontsize=13, fontweight="bold", pad=15)
plt.xlabel("Hours Studied", fontsize=11, labelpad=10)
plt.ylabel("Probability of Passing", fontsize=11, labelpad=10)
plt.ylim(-0.05, 1.05)
plt.xlim(0.5, 10.5)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="center left", frameon=True, facecolor="white", edgecolor="#E5E7EB")
plt.tight_layout()

# Save the plot
plot_path = "plots/examples_2_classification.png"
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"\nVisual plot successfully saved to: {plot_path}")
print("====================================================")

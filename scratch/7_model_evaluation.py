#!/usr/bin/env python3
import math
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

"""
Model Evaluation Metrics from Scratch (No Frameworks)
=====================================================

Example Used:
-------------
We want to evaluate prediction accuracy for two models:
  - **Part 1: Classification Model** (evaluating 10 patient diagnostic outcomes):
    - Actual Target Outcomes: [0, 1, 1, 0, 1, 1, 0, 0, 1, 0] (1 = Disease, 0 = Healthy)
    - Predicted Model Outputs: [0, 1, 0, 0, 1, 1, 1, 0, 1, 0] (Model prediction classifications)

  - **Part 2: Regression Model** (evaluating 5 house price forecasts):
    - Actual Target Prices: [300, 450, 200, 600, 350] (in thousands of dollars)
    - Predicted Model Prices: [280, 470, 180, 550, 370] (Model price predictions)

Mathematical Goal:
------------------
1. **Classification Metrics**:
   - Compute True Positive (TP), False Positive (FP), True Negative (TN), and False Negative (FN).
   - Accuracy  = (TP + TN) / Total
   - Precision = TP / (TP + FP)
   - Recall    = TP / (TP + FN)
   - F1-Score  = 2 * (Precision * Recall) / (Precision + Recall)
2. **Regression Metrics**:
   - MAE  = sum(|actual - predicted|) / n
   - MSE  = sum((actual - predicted)^2) / n
   - RMSE = sqrt(MSE)
   - R²   = 1 - (sum((actual - predicted)^2) / sum((actual - mean_actual)^2))
"""

# -----------------------------------------------------------
# PART 1: Classification Metrics Evaluation
# -----------------------------------------------------------
print("====================================================")
print("PART 1: Classification Evaluation (Diagnostic Diagnostics)")

# Ground Truth outcomes vs Predicted outcomes
y_true = [0, 1, 1, 0, 1, 1, 0, 0, 1, 0]
y_pred = [0, 1, 0, 0, 1, 1, 1, 0, 1, 0]
m_class = len(y_true)

# Calculate Confusion Matrix counts
tp = 0
fp = 0
tn = 0
fn = 0

for actual, prediction in zip(y_true, y_pred):
    if actual == 1 and prediction == 1:
        tp += 1
    elif actual == 0 and prediction == 1:
        fp += 1
    elif actual == 0 and prediction == 0:
        tn += 1
    elif actual == 1 and prediction == 0:
        fn += 1

# Calculate standard metrics
accuracy = (tp + tn) / m_class

# Handle potential division-by-zero errors in case no positive alerts occur
precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

print(f"\nConfusion Matrix details:")
print(f" - True Negatives (TN): {tn} (Correctly classified as Healthy)")
print(f" - False Positives (FP): {fp} (Healthy classified as Disease - Type I Error)")
print(f" - False Negatives (FN): {fn} (Disease classified as Healthy - Type II Error)")
print(f" - True Positives (TP): {tp} (Correctly classified as Disease)")

print(f"\nCalculated Classification Metrics:")
print(f" - Accuracy : {accuracy * 100:.2f}%")
print(f" - Precision: {precision * 100:.2f}%")
print(f" - Recall   : {recall * 100:.2f}%")
print(f" - F1-Score : {f1 * 100:.2f}%")

# Record confusion matrix details for plotting
print("\nConfusion Matrix details recorded for visual plotting.")


# -----------------------------------------------------------
# PART 2: Regression Metrics Evaluation
# -----------------------------------------------------------
print("\n====================================================")
print("PART 2: Regression Evaluation (House Price Predictions)")

# Actual Target Prices vs Predicted Model Prices
prices_true = [300, 450, 200, 600, 350]
prices_pred = [280, 470, 180, 550, 370]
n_reg = len(prices_true)

# Calculate sum of differences
sum_absolute_error = 0.0
sum_squared_error = 0.0

for actual, prediction in zip(prices_true, prices_pred):
    error = actual - prediction
    sum_absolute_error += abs(error)
    sum_squared_error += error ** 2

# Calculate MAE, MSE, RMSE
mae = sum_absolute_error / n_reg
mse = sum_squared_error / n_reg
rmse = math.sqrt(mse)

# Calculate R-squared (coefficient of determination)
mean_actual = sum(prices_true) / n_reg
sum_total_variance = sum((actual - mean_actual)**2 for actual in prices_true)
r_squared = 1.0 - (sum_squared_error / sum_total_variance)

print(f"\nCalculated Regression Metrics:")
print(f" - Mean Absolute Error (MAE) : {mae:.2f}")
print(f" - Mean Squared Error (MSE)   : {mse:.2f}")
print(f" - Root Mean Sq Error (RMSE)  : ${rmse:.2f}k (avg distance from regression line)")
print(f" - R² (Variance Explained)    : {r_squared * 100:.2f}%")
# 5. Generate and save visualization plots (Subplots: Left = Conf Matrix, Right = Prices Residuals)
os.makedirs("plots", exist_ok=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: Confusion Matrix Heatmap from Scratch values
cm = [[tn, fp], [fn, tp]]
im = ax1.imshow(cm, cmap="Blues", interpolation="nearest")
ax1.set_title("Confusion Matrix (From Scratch)", fontsize=12, fontweight="bold", pad=10)
fig.colorbar(im, ax=ax1, shrink=0.7)

classes = ["Healthy (0)", "Disease (1)"]
ax1.set_xticks([0, 1])
ax1.set_xticklabels(classes)
ax1.set_yticks([0, 1])
ax1.set_yticklabels(classes)

# Annotate counts inside the cells
max_val = max(tn, fp, fn, tp)
thresh = max_val / 2.0
for i in range(2):
    for j in range(2):
        val = cm[i][j]
        label = "TN" if i==0 and j==0 else "FP" if i==0 and j==1 else "FN" if i==1 and j==0 else "TP"
        ax1.text(j, i, f"Count: {val}\n({label})",
                 horizontalalignment="center",
                 color="white" if val > thresh else "black",
                 fontsize=11, fontweight="bold")

ax1.set_ylabel("Actual Label", fontsize=11)
ax1.set_xlabel("Predicted Label", fontsize=11)

# Subplot 2: Actual vs Predicted Prices Grouped Bar Chart
x_indices = list(range(len(prices_true)))
bar_width = 0.35

ax2.bar([x - bar_width/2 for x in x_indices], prices_true, bar_width, color="#4F46E5", label="Actual Price")
ax2.bar([x + bar_width/2 for x in x_indices], prices_pred, bar_width, color="#10B981", label="Predicted Price")

# Draw residual lines
for i in range(len(prices_true)):
    ax2.plot([i, i], [prices_true[i], prices_pred[i]], color="#EF4444", linestyle="--", linewidth=1.5)

ax2.set_xlabel("House Sample Index", fontsize=11)
ax2.set_ylabel("Price ($ thousands)", fontsize=11)
ax2.set_title("Actual vs. Predicted House Prices (Residuals)", fontsize=12, fontweight="bold", pad=10)
ax2.set_xticks(x_indices)
ax2.set_xticklabels([f"House {i+1}" for i in range(len(prices_true))])
ax2.grid(True, axis="y", linestyle="--", alpha=0.5)
ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#E5E7EB")

plt.suptitle("Model Evaluation from Scratch: Metrics & Price Residuals", fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout()

# Save the plot
plot_path = "plots/scratch_7_evaluation.png"
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"\nVisual plot successfully saved to: {plot_path}")
print("====================================================")

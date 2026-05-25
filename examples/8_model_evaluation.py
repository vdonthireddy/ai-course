#!/usr/bin/env python3
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, 
    recall_score, f1_score, mean_squared_error, r2_score
)

print("====================================================")
print("PART 1: Classification Metrics Evaluation")
# 1. Classification Target & Predictions
# Imagine 10 patients: 1 = Has disease, 0 = Healthy
y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 0])
y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 0, 1, 0]) # Prediction model output

print(f" Actual Labels   : {y_true}")
print(f" Predicted Labels: {y_pred}")

# 2. Compute confusion matrix and metrics
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\nConfusion Matrix:")
print(f"                  Predicted Healthy    Predicted Disease")
print(f" Actual Healthy:        TN: {tn}                 FP: {fp}")
print(f" Actual Disease:        FN: {fn}                 TP: {tp}")

print(f"\nCalculated Classification Metrics:")
print(f" - Accuracy  : {accuracy * 100:.1f}%  (TP+TN) / Total")
print(f" - Precision : {precision * 100:.1f}%  TP / (TP+FP) - quality of disease alarms")
print(f" - Recall    : {recall * 100:.1f}%  TP / (TP+FN) - coverage of actual sick patients")
print(f" - F1-Score  : {f1 * 100:.1f}%  Harmonic mean of Precision & Recall")


print("\n====================================================")
print("PART 2: Regression Metrics Evaluation")
# 3. Regression Target & Predictions
# Imagine 5 home price values (actual in $k vs model predictions)
prices_true = np.array([300, 450, 200, 600, 350])
prices_pred = np.array([280, 470, 180, 550, 370])

print(f" Actual Prices   : {prices_true}")
print(f" Predicted Prices: {prices_pred}")

mse = mean_squared_error(prices_true, prices_pred)
rmse = np.sqrt(mse)
r2 = r2_score(prices_true, prices_pred)

print(f"\nCalculated Regression Metrics:")
print(f" - MSE (Mean Squared Error) : {mse:.2f}")
print(f" - RMSE (Root Mean Sq Error): ${rmse:.2f}k  (average error deviation)")
print(f" - R² (Variance Explained)  : {r2 * 100:.1f}%")

# 4. Generate and save visualization plots (Subplots: Left = Conf Matrix Heatmap, Right = Price Predictions Residuals)
os.makedirs("plots", exist_ok=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: Confusion Matrix Heatmap
cm = confusion_matrix(y_true, y_pred)
im = ax1.imshow(cm, cmap="Blues", interpolation="nearest")
ax1.set_title("Classification Confusion Matrix", fontsize=12, fontweight="bold", pad=10)
fig.colorbar(im, ax=ax1, shrink=0.7)

classes = ["Healthy (0)", "Disease (1)"]
tick_marks = np.arange(len(classes))
ax1.set_xticks(tick_marks)
ax1.set_xticklabels(classes)
ax1.set_yticks(tick_marks)
ax1.set_yticklabels(classes)

# Annotate cell counts
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax1.text(j, i, f"Count: {cm[i, j]}\n" + ("(TP)" if i==1 and j==1 else "(TN)" if i==0 and j==0 else "(FP)" if i==0 and j==1 else "(FN)"),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=11, fontweight="bold")

ax1.set_ylabel("Actual Label", fontsize=11)
ax1.set_xlabel("Predicted Label", fontsize=11)

# Subplot 2: Actual vs Predicted Prices Grouped Bar Chart
x_indices = np.arange(len(prices_true))
bar_width = 0.35

ax2.bar(x_indices - bar_width/2, prices_true, bar_width, color="#4F46E5", label="Actual Price")
ax2.bar(x_indices + bar_width/2, prices_pred, bar_width, color="#10B981", label="Predicted Price")

# Draw error line segments connecting them (residuals)
for i in range(len(prices_true)):
    ax2.plot([i, i], [prices_true[i], prices_pred[i]], color="#EF4444", linestyle="--", linewidth=1.5)

ax2.set_xlabel("House Sample Index", fontsize=11)
ax2.set_ylabel("Price ($ thousands)", fontsize=11)
ax2.set_title("Actual vs. Predicted House Prices (Residuals)", fontsize=12, fontweight="bold", pad=10)
ax2.set_xticks(x_indices)
ax2.set_xticklabels([f"House {i+1}" for i in range(len(prices_true))])
ax2.grid(True, axis="y", linestyle="--", alpha=0.5)
ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#E5E7EB")

plt.suptitle("Model Evaluation: Classification & Regression Comparison", fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout()

# Save the plot
plot_path = "plots/examples_8_evaluation.png"
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"\nVisual plot successfully saved to: {plot_path}")
print("====================================================")

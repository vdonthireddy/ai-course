#!/usr/bin/env python3
import math

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

# Output ASCII confusion matrix grid layout
print("\nVisual Confusion Matrix Grid:")
print("                   Predicted Healthy   Predicted Disease")
print(f"  Actual Healthy:    [ TN = {tn} ]          [ FP = {fp} ]")
print(f"  Actual Disease:    [ FN = {fn} ]          [ TP = {tp} ]")


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
print("====================================================")

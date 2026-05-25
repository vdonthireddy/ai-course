#!/usr/bin/env python3
import numpy as np
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
print("====================================================")

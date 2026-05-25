#!/usr/bin/env python3
import numpy as np
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
print("====================================================")

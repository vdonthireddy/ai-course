#!/usr/bin/env python3
import numpy as np
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
print("====================================================")

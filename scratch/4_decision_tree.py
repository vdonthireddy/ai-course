#!/usr/bin/env python3
"""
Decision Tree Split (Decision Stump) from Scratch (No Frameworks)
================================================================

Example Used:
-------------
We want to predict whether a customer will churn (1 = Churned, 0 = Loyal) based on:
  - Age
  - Support Calls (number of times they called support in the last month)

Dataset of 10 customers:
  - Cust 1: Age = 25, Support Calls = 8 -> Churn (1)
  - Cust 2: Age = 45, Support Calls = 1 -> Loyal (0)
  - Cust 3: Age = 30, Support Calls = 4 -> Loyal (0)
  - Cust 4: Age = 22, Support Calls = 9 -> Churn (1)
  - Cust 5: Age = 55, Support Calls = 0 -> Loyal (0)
  - Cust 6: Age = 35, Support Calls = 2 -> Loyal (0)
  - Cust 7: Age = 28, Support Calls = 6 -> Churn (1)
  - Cust 8: Age = 40, Support Calls = 3 -> Loyal (0)
  - Cust 9: Age = 50, Support Calls = 1 -> Loyal (0)
  - Cust 10: Age = 26, Support Calls = 7 -> Churn (1)

Mathematical Goal:
------------------
We build a **Decision Stump** (a decision tree with exactly 1 split). 
We find the split threshold that minimizes **Gini Impurity**:
  Gini = 1 - (p_0^2 + p_1^2)
Where p_0 and p_1 are the proportions of Loyal (0) and Churned (1) customers in a node.
For any split, the total impurity is the weighted average Gini of the left and right child nodes:
  Total_Gini = (n_left * Gini_left + n_right * Gini_right) / n_total
"""

# 1. Dataset
# Features: [Age, Support Calls]
X = [
    [25, 8], [45, 1], [30, 4], [22, 9], [55, 0],
    [35, 2], [28, 6], [40, 3], [50, 1], [26, 7]
]
# Targets: 0 = Loyal, 1 = Churned
y = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1]
features_names = ["Age", "Support Calls"]

print("====================================================")
print("Customer Support Dataset:")
for idx, (feat, label) in enumerate(zip(X, y)):
    status = "Churned" if label == 1 else "Loyal"
    print(f" - Cust {idx+1:2d}: Age: {feat[0]}, Support Calls: {feat[1]} -> {status}")

# 2. Gini Impurity calculation
def calculate_gini(labels):
    n = len(labels)
    if n == 0:
        return 0.0
    
    # Count occurrences of each class
    count_0 = sum(1 for label in labels if label == 0)
    count_1 = n - count_0
    
    p0 = count_0 / n
    p1 = count_1 / n
    
    # Gini formula
    return 1.0 - (p0**2 + p1**2)

# Calculate base Gini before any split
base_gini = calculate_gini(y)
print(f"\nInitial Dataset Gini Impurity: {base_gini:.4f}")

# 3. Splitting Search Loop
best_gini = 999.0
best_feature_idx = -1
best_threshold = -1.0
best_left_prediction = -1
best_right_prediction = -1

# Search through all features (0 = Age, 1 = Support Calls)
for feature_idx in range(2):
    # Find unique values to test as split boundaries
    feature_values = sorted(list(set(row[feature_idx] for row in X)))
    
    # Try midpoints between adjacent sorted values as potential thresholds
    for i in range(len(feature_values) - 1):
        threshold = (feature_values[i] + feature_values[i+1]) / 2.0
        
        # Split targets based on threshold
        left_labels = [y[j] for j in range(len(X)) if X[j][feature_idx] <= threshold]
        right_labels = [y[j] for j in range(len(X)) if X[j][feature_idx] > threshold]
        
        # Calculate individual impurities
        gini_left = calculate_gini(left_labels)
        gini_right = calculate_gini(right_labels)
        
        # Calculate weighted average impurity
        weighted_gini = (len(left_labels) * gini_left + len(right_labels) * gini_right) / len(y)
        
        # We want to find the split that minimizes Gini (maximized information gain)
        if weighted_gini < best_gini:
            best_gini = weighted_gini
            best_feature_idx = feature_idx
            best_threshold = threshold
            
            # Predict the majority class in each branch
            best_left_prediction = 1 if sum(left_labels) > len(left_labels)/2.0 else 0
            best_right_prediction = 1 if sum(right_labels) > len(right_labels)/2.0 else 0

# 4. Display Split Rule
print("\n--- Split Optimization Result ---")
feat_name = features_names[best_feature_idx]
print(f" Best Split Feature: '{feat_name}'")
print(f" Best Split Threshold: {best_threshold}")
print(f" Weighted Gini Impurity after split: {best_gini:.4f}")

# 5. Tree Prediction Logic
def predict(point):
    val = point[best_feature_idx]
    if val <= best_threshold:
        return best_left_prediction
    else:
        return best_right_prediction

# 6. Test Predictions
print("\nModel Prediction Rules:")
left_label = "Churned" if best_left_prediction == 1 else "Loyal"
right_label = "Churned" if best_right_prediction == 1 else "Loyal"
print(f" - IF '{feat_name}' <= {best_threshold} THEN Predict: {left_label}")
print(f" - IF '{feat_name}' > {best_threshold} THEN Predict: {right_label}")

# Test Case: Customer with Age 24, 6 support calls
test_customer = [24, 6]
prediction = predict(test_customer)
predicted_status = "Churned" if prediction == 1 else "Loyal"
print(f"\nPrediction for new customer (Age: 24, Support Calls: 6): {predicted_status}")

# Output ASCII Decision Tree structure
print("\nVisual Decision Tree Diagram:")
print("                 [ Is Customer Age <= 29.0? ]")
print("                       /              \\")
print("                     Yes              No")
print("                     /                  \\")
print("             [ Predict: Churned ]   [ Predict: Loyal ]")
print("====================================================")

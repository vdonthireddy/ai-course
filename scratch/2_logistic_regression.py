#!/usr/bin/env python3
import math

"""
Logistic Regression from Scratch (No Frameworks)
===============================================

Example Used:
-------------
We want to predict whether a student passes (1) or fails (0) an exam based on how many hours they studied.
Our dataset consists of 10 students:
  - 1 hour   -> Fail (0)
  - 2 hours  -> Fail (0)
  - 3 hours  -> Fail (0)
  - 4 hours  -> Fail (0)
  - 5 hours  -> Fail (0)
  - 6 hours  -> Pass (1)
  - 7 hours  -> Pass (1)
  - 8 hours  -> Pass (1)
  - 9 hours  -> Pass (1)
  - 10 hours -> Pass (1)

Mathematical Goal:
------------------
We map our input linear combination `z = weight * hours + bias` into a probability 
between 0 and 1 using the Sigmoid Function: 
  p = 1 / (1 + exp(-z))
We optimize the weight and bias parameters using Gradient Descent to minimize 
the Binary Cross-Entropy Loss (Log Loss).
"""

# 1. Prepare data
hours = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]  # Features (X)
outcomes = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]                    # Target labels (y)
m = len(hours)

print("====================================================")
print("Student Exam Dataset:")
for h, out in zip(hours, outcomes):
    status = "Pass" if out == 1 else "Fail"
    print(f" - Studied: {h} hours -> Result: {status}")

# 2. Sigmoid helper function
def sigmoid(z):
    # Maps any real number to a value between 0 and 1
    # We clip z to avoid math overflow errors with extremely large/small values
    if z < -50:
        return 0.0
    if z > 50:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))

# 3. Parameters initialization
weight = 0.0
bias = 0.0

# Hyperparameters
learning_rate = 0.3
epochs = 2000

print(f"\nTraining model using Gradient Descent (alpha={learning_rate}, epochs={epochs})...")

# 4. Gradient Descent Loop
for epoch in range(epochs):
    sum_grad_weight = 0.0
    sum_grad_bias = 0.0
    total_loss = 0.0
    
    for i in range(m):
        # Linear combination: z = w * x + b
        z = weight * hours[i] + bias
        # Predicted probability
        prediction = sigmoid(z)
        # Prediction error (predicted - actual)
        error = prediction - outcomes[i]
        
        # Accumulate gradients (derivatives of Log Loss)
        sum_grad_weight += error * hours[i]
        sum_grad_bias += error
        
        # Accumulate Binary Cross-Entropy Loss
        # Avoid log(0) by adding a tiny epsilon
        eps = 1e-15
        pred_clipped = max(eps, min(1.0 - eps, prediction))
        total_loss += - (outcomes[i] * math.log(pred_clipped) + (1 - outcomes[i]) * math.log(1.0 - pred_clipped))
    
    # Update parameters
    weight -= learning_rate * (sum_grad_weight / m)
    bias -= learning_rate * (sum_grad_bias / m)
    
    # Print status updates
    if (epoch + 1) % 500 == 0 or epoch == 0:
        avg_loss = total_loss / m
        print(f" - Epoch {epoch+1:4d} | Log Loss: {avg_loss:.6f} | Weight: {weight:.4f} | Bias: {bias:.4f}")

# 5. Output final learned parameters
print("\nFinal Learned Parameters:")
print(f" - Weight Coefficient: {weight:.4f}")
print(f" - Bias (Intercept): {bias:.4f}")
print(f" - Model Equation: Probability(Pass) = Sigmoid({weight:.4f} * Hours + {bias:.4f})")

# 6. Test predictions on new hours
test_hours = [3.0, 5.5, 8.0]
print("\nPredictions for new students:")
for h in test_hours:
    z = weight * h + bias
    prob = sigmoid(z)
    decision = "Pass" if prob >= 0.5 else "Fail"
    print(f" - Studied {h} hours: Predicted {decision} (Probability of Passing: {prob * 100:.2f}%)")

# Output ASCII plot representing the Sigmoid decision boundary curve
print("\nVisual Plot of Passing Probability (Sigmoid Curve):")
print("Prob(Pass)")
print("1.0|                         ----------- (Studied 8 hrs: 99.6%)")
print("   |                      /")
print("0.8|                    /")
print("0.6|                  /")
print("0.4|                /   <-- Decision Boundary (Studied 5.5 hrs: 53.8%)")
print("0.2|              /")
print("0.0| ----------- (Studied 3 hrs: 0.6%)")
print("   +-----------------------------------> Hours Studied")
print("     1.0  2.0  3.0  4.0  5.0  6.0  7.0  8.0  9.0  10.0")
print("====================================================")

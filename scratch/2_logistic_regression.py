#!/usr/bin/env python3
import math
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

# Track loss history for plotting
loss_history = []

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
    
    avg_loss = total_loss / m
    loss_history.append(avg_loss)
    
    # Print status updates
    if (epoch + 1) % 500 == 0 or epoch == 0:
        print(f" - Epoch {epoch+1:4d} | Log Loss: {avg_loss:.6f} | Weight: {weight:.4f} | Bias: {bias:.4f}")

# 5. Output final learned parameters
print("\nFinal Learned Parameters:")
print(f" - Weight Coefficient: {weight:.4f}")
print(f" - Bias (Intercept): {bias:.4f}")
print(f" - Model Equation: Probability(Pass) = Sigmoid({weight:.4f} * Hours + {bias:.4f})")

# 6. Test predictions on new hours
test_hours = [3.0, 5.5, 8.0]
print("\nPredictions for new students:")
test_probs = []
for h in test_hours:
    z = weight * h + bias
    prob = sigmoid(z)
    test_probs.append(prob)
    decision = "Pass" if prob >= 0.5 else "Fail"
    print(f" - Studied {h} hours: Predicted {decision} (Probability of Passing: {prob * 100:.2f}%)")

# 7. Generate and save visualization plot (Subplots: Left = Sigmoid Fit, Right = Loss Curve)
os.makedirs("plots", exist_ok=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Helper to generate numbers from-scratch
def linspace(start, stop, num):
    return [start + (stop - start) * i / (num - 1) for i in range(num)]

# Subplot 1: Logistic Curve Fit
fail_hours = [hours[i] for i in range(m) if outcomes[i] == 0]
pass_hours = [hours[i] for i in range(m) if outcomes[i] == 1]
ax1.scatter(fail_hours, [0] * len(fail_hours), color="#EF4444", s=80, marker="o", edgecolor="black", label="Actual Fail (0)", zorder=3)
ax1.scatter(pass_hours, [1] * len(pass_hours), color="#10B981", s=80, marker="o", edgecolor="black", label="Actual Pass (1)", zorder=3)

# Plot smooth sigmoid line
hours_line = linspace(0.5, 10.5, 200)
probs_line = [sigmoid(weight * h + bias) for h in hours_line]
ax1.plot(hours_line, probs_line, color="#4F46E5", linewidth=3, label="Sigmoid Probability Curve", zorder=2)

# Highlight test prediction points
ax1.scatter(test_hours, test_probs, color="#F59E0B", marker="*", s=250, edgecolor="black", label="Predictions for New Students", zorder=5)

# Draw decision boundary (50% probability threshold)
db_hours = -bias / weight
ax1.axvline(x=db_hours, color="#6B7280", linestyle="--", alpha=0.8, label=f"Decision Boundary ({db_hours:.2f} hrs)")

ax1.set_xlabel("Hours Studied", fontsize=11, labelpad=10)
ax1.set_ylabel("Probability of Passing", fontsize=11, labelpad=10)
ax1.set_title("Logistic Sigmoid Regression Fit (From Scratch)", fontsize=12, fontweight="bold")
ax1.set_ylim(-0.05, 1.05)
ax1.set_xlim(0.5, 10.5)
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(loc="center left", frameon=True, facecolor="white", edgecolor="#E5E7EB")

# Subplot 2: Log Loss Decay Curve
ax2.plot(range(1, epochs + 1), loss_history, color="#EF4444", linewidth=2.5, label="Log Loss Decay")
ax2.set_xlabel("Epochs", fontsize=11, labelpad=10)
ax2.set_ylabel("Binary Cross-Entropy (Log) Loss", fontsize=11, labelpad=10)
ax2.set_title("Gradient Descent Optimization Convergence", fontsize=12, fontweight="bold")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#E5E7EB")

plt.suptitle("Logistic Regression & Binary Cross-Entropy Loss from Scratch", fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout()

# Save the plot
plot_path = "plots/scratch_2_logistic.png"
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"\nVisual plot successfully saved to: {plot_path}")
print("====================================================")

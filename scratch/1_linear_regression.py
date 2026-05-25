#!/usr/bin/env python3
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

"""
Simple Linear Regression from Scratch (No Frameworks)
=====================================================

Example Used:
-------------
We want to predict a house's price (in thousands of dollars) based on its size (in square feet).
We have a training dataset of 5 houses:
  - House 1: Size = 1000 sq ft -> Price = $200k
  - House 2: Size = 1500 sq ft -> Price = $300k
  - House 3: Size = 2000 sq ft -> Price = $400k
  - House 4: Size = 2500 sq ft -> Price = $500k
  - House 5: Size = 3000 sq ft -> Price = $600k

Mathematical Goal:
------------------
We want to find the parameters for the line: Price = Weight * Size + Bias
Here, the actual relation is exactly: Price = 0.2 * Size + 0 (i.e. $200 per sq ft).
We will use Gradient Descent to learn the Weight and Bias from scratch.
"""

# 1. Define dataset
sizes = [1000, 1500, 2000, 2500, 3000] # Input feature (X)
prices = [200, 300, 400, 500, 600]     # Target label (y)
m = len(sizes)                         # Number of samples

# To help gradient descent converge quickly and prevent numerical overflow, 
# we scale the features (Size / 1000). For example, 1500 sq ft becomes 1.5.
# This is called feature scaling.
X = [s / 1000.0 for s in sizes]
y = prices

print("====================================================")
print("Training Data (Scaled size vs Price):")
for raw_s, scaled_s, p in zip(sizes, X, y):
    print(f" - Size: {raw_s} sq ft (scaled: {scaled_s}) -> Price: ${p}k")

# 2. Initialize parameters
weight = 0.0  # Initial weight
bias = 0.0    # Initial bias (intercept)

# Hyperparameters
learning_rate = 0.05
epochs = 1000

print(f"\nTraining model using Gradient Descent (alpha={learning_rate}, epochs={epochs})...")

# Track MSE Loss history for plotting
loss_history = []

# 3. Gradient Descent Loop
for epoch in range(epochs):
    sum_error_weight = 0.0
    sum_error_bias = 0.0
    total_squared_error = 0.0
    
    # Calculate gradients across all samples
    for i in range(m):
        prediction = weight * X[i] + bias
        error = prediction - y[i]
        
        # Accumulate gradients (derivatives of MSE)
        sum_error_weight += error * X[i]
        sum_error_bias += error
        
        # Accumulate squared error for monitoring loss
        total_squared_error += error ** 2
    
    # Average the gradients
    grad_weight = sum_error_weight / m
    grad_bias = sum_error_bias / m
    
    # Update weight and bias parameters
    weight -= learning_rate * grad_weight
    bias -= learning_rate * grad_bias
    
    # Average MSE loss
    mse = total_squared_error / (2 * m)
    loss_history.append(mse)
    
    # Print status every 200 epochs
    if (epoch + 1) % 200 == 0 or epoch == 0:
        print(f" - Epoch {epoch+1:4d} | MSE Loss: {mse:10.4f} | Weight: {weight:.4f} | Bias: {bias:.4f}")

# 4. Predictions on new data
print("\nFinal Learned Parameters:")
# Remember to divide weight by 1000 because we trained on scaled sizes (sizes/1000)
final_weight_unscaled = weight / 1000.0
print(f" - Weight (Price per sq ft): ${final_weight_unscaled * 1000:.2f} (i.e. {final_weight_unscaled:.4f} per sq ft)")
print(f" - Bias (Base price intercept): ${bias:.2f}k")
print(f" - Model Equation: Price = {final_weight_unscaled:.4f} * Size + {bias:.2f}")

# Test predictions
test_sizes = [1200, 2200, 3500]
print("\nPredictions for new houses:")
test_predictions = []
for size in test_sizes:
    # Scale input first
    scaled_size = size / 1000.0
    predicted_price = weight * scaled_size + bias
    test_predictions.append(predicted_price)
    print(f" - A {size} sq ft house is predicted to cost: ${predicted_price:.2f}k")

# 5. Generate and save visualization plot (Subplots: Left = Fit Line, Right = Loss Curve)
os.makedirs("plots", exist_ok=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Helper to generate numbers from-scratch
def linspace(start, stop, num):
    return [start + (stop - start) * i / (num - 1) for i in range(num)]

# Subplot 1: Fitted Regression Line vs Actual Data
ax1.scatter(sizes, prices, color="#4F46E5", s=100, label="Actual Data Points", zorder=3)
# Draw prediction line
sizes_line = linspace(800, 3200, 100)
# Wait, sizes_line scaled is sizes_line / 1000.0
prices_line = [ (s / 1000.0) * weight + bias for s in sizes_line ]
ax1.plot(sizes_line, prices_line, color="#10B981", linewidth=3, label=f"Fitted Line (Price = {final_weight_unscaled:.4f}*Size + {bias:.2f})")

# Highlight test prediction points
ax1.scatter(test_sizes, test_predictions, color="#F59E0B", marker="*", s=250, edgecolor="black", label="Predictions for New Houses", zorder=5)

ax1.set_xlabel("House Size (sq ft)", fontsize=11, labelpad=10)
ax1.set_ylabel("Price ($ thousands)", fontsize=11, labelpad=10)
ax1.set_title("Fitted Linear Regression Line (From Scratch)", fontsize=12, fontweight="bold")
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#E5E7EB")

# Subplot 2: MSE Loss Decay Curve
ax2.plot(range(1, epochs + 1), loss_history, color="#EF4444", linewidth=2.5, label="MSE Loss Decay")
ax2.set_xlabel("Epochs", fontsize=11, labelpad=10)
ax2.set_ylabel("Mean Squared Error (MSE) Loss", fontsize=11, labelpad=10)
ax2.set_title("Gradient Descent Optimization Convergence", fontsize=12, fontweight="bold")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#E5E7EB")

plt.suptitle("Simple Linear Regression & Gradient Descent from Scratch", fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout()

# Save the plot
plot_path = "plots/scratch_1_regression.png"
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"\nVisual plot successfully saved to: {plot_path}")
print("====================================================")

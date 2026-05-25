#!/usr/bin/env python3
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
    
    # Print status every 200 epochs
    if (epoch + 1) % 200 == 0 or epoch == 0:
        mse = total_squared_error / (2 * m)
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
for size in test_sizes:
    # Scale input first
    scaled_size = size / 1000.0
    predicted_price = weight * scaled_size + bias
    print(f" - A {size} sq ft house is predicted to cost: ${predicted_price:.2f}k")

# Output ASCII plot representing the linear regression line vs data points
print("\nVisual Plot of Fitted Regression Line:")
print("Price ($k)")
print("  ^")
print("800|                                   # (Actual 3.0k, 600)")
print("   |                                 /")
print("600|                       # (Actual 2.5k, 500)")
print("   |                     /")
print("400|             # (Actual 2.0k, 400)")
print("   |           /")
print("200|   # (Actual 1.0k, 200)")
print("   | /")
print("  0+-----------------------------------> Size (x1000 sq ft)")
print("    1.0   1.5   2.0   2.5   3.0   3.5   4.0")
print("   Legend: '#' = Actual Data Points, '/' = Fitted Regression Line")
print("====================================================")

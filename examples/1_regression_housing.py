#!/usr/bin/env python3
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split

print("====================================================")
# 1. Prepare data (House size in sq ft vs Price in thousands of dollars)
# Example: 1500 sq ft house roughly $290k
X = np.array([[1000], [1200], [1500], [1800], [2000], [2400], [2800], [3000], [3500], [4000]])
y = np.array([200, 230, 290, 340, 380, 470, 530, 580, 680, 790])

print("Housing Dataset (Features = Square Footage, Target = Price in $k):")
for size, price in zip(X.flatten(), y):
    print(f" - Size: {size} sq ft -> Price: ${price}k")

# 2. Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Fit Ordinary Least Squares (OLS) Linear Regression
ols = LinearRegression()
ols.fit(X_train, y_train)

# 4. Fit Ridge (L2 Regularized) and Lasso (L1 Regularized) Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

lasso = Lasso(alpha=10.0)
lasso.fit(X_train, y_train)

# 5. Output results
print("\nModel Parameters & Coefficients:")
print(f" OLS Line  : Price = {ols.intercept_:.2f} + {ols.coef_[0]:.4f} * Size")
print(f" Ridge Line: Price = {ridge.intercept_:.2f} + {ridge.coef_[0]:.4f} * Size")
print(f" Lasso Line: Price = {lasso.intercept_:.2f} + {lasso.coef_[0]:.4f} * Size")

# 6. Predict value for a new house (e.g. 2200 sq ft)
new_house_size = np.array([[2200]])
ols_pred = ols.predict(new_house_size)[0]
ridge_pred = ridge.predict(new_house_size)[0]
lasso_pred = lasso.predict(new_house_size)[0]

print(f"\nPredictions for a new 2,200 sq ft house:")
print(f" - OLS prediction  : ${ols_pred:.2f}k")
print(f" - Ridge prediction: ${ridge_pred:.2f}k")
print(f" - Lasso prediction: ${lasso_pred:.2f}k")

# 7. Generate and save visualization plot
os.makedirs("plots", exist_ok=True)
plt.figure(figsize=(9, 6))

# Plot actual data points
plt.scatter(X, y, color="#4F46E5", s=80, label="Actual Data Points (Dataset)", zorder=3)

# Generate fit line values
X_range = np.linspace(900, 4100, 100).reshape(-1, 1)
plt.plot(X_range, ols.predict(X_range), color="#10B981", linestyle="-", linewidth=2.5, label="OLS Regression Line")
plt.plot(X_range, ridge.predict(X_range), color="#F59E0B", linestyle="--", linewidth=2, label="Ridge Regression (L2)")
plt.plot(X_range, lasso.predict(X_range), color="#EF4444", linestyle=":", linewidth=2, label="Lasso Regression (L1)")

# Highlight 2200 sq ft prediction points
plt.scatter([2200], [ols_pred], color="#10B981", marker="*", s=250, edgecolor="black", zorder=5, label="OLS Prediction at 2,200 sq ft")
plt.scatter([2200], [ridge_pred], color="#F59E0B", marker="s", s=100, edgecolor="black", zorder=5, label="Ridge Prediction at 2,200 sq ft")
plt.scatter([2200], [lasso_pred], color="#EF4444", marker="^", s=100, edgecolor="black", zorder=5, label="Lasso Prediction at 2,200 sq ft")

plt.title("Housing Prices vs. House Size (OLS, Ridge, and Lasso Comparison)", fontsize=13, fontweight="bold", pad=15)
plt.xlabel("House Size (sq ft)", fontsize=11, labelpad=10)
plt.ylabel("Price ($ thousands)", fontsize=11, labelpad=10)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#E5E7EB")
plt.tight_layout()

# Save the plot
plot_path = "plots/examples_1_regression.png"
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"\nVisual plot successfully saved to: {plot_path}")
print("====================================================")


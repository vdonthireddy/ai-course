#!/usr/bin/env python3
import numpy as np
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
print("====================================================")

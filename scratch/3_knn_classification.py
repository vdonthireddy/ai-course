#!/usr/bin/env python3
import math

"""
K-Nearest Neighbors (KNN) from Scratch (No Frameworks)
=====================================================

Example Used:
-------------
We want to predict if a new customer accepts a credit card offer (1 = Accepted, 0 = Declined) 
based on two features: Age and Annual Income (in $k).
We have 10 existing customer profiles:
  - Cust 1: Age = 22, Income = $35k -> Declined (0)
  - Cust 2: Age = 25, Income = $40k -> Declined (0)
  - Cust 3: Age = 45, Income = $120k -> Accepted (1)
  - Cust 4: Age = 50, Income = $150k -> Accepted (1)
  - Cust 5: Age = 30, Income = $60k -> Declined (0)
  - Cust 6: Age = 32, Income = $55k -> Declined (0)
  - Cust 7: Age = 40, Income = $90k -> Accepted (1)
  - Cust 8: Age = 28, Income = $48k -> Declined (0)
  - Cust 9: Age = 55, Income = $130k -> Accepted (1)
  - Cust 10: Age = 21, Income = $30k -> Declined (0)

Mathematical Goal:
------------------
1. **Feature Scaling (Standardization)**:
   KNN relies on distance. Since Income values (30 to 150) are much larger than Age values (21 to 55), 
   Income would dominate the distance calculation if we didn't scale them. 
   We standardize features by subtracting the mean and dividing by the standard deviation: 
     z = (x - mean) / std_dev
2. **Euclidean Distance**:
   Calculate distance from query point to all training points:
     distance = sqrt((x1_scaled - x2_scaled)^2 + (y1_scaled - y2_scaled)^2)
3. **Voting**:
   Select the top K closest neighbors and count votes to make the final prediction.
"""

# 1. Dataset
# Features: [Age, Income]
X = [
    [22, 35], [25, 40], [45, 120], [50, 150], [30, 60],
    [32, 55], [40, 90],  [28, 48],  [55, 130], [21, 30]
]
# Target labels: 0 = Declined, 1 = Accepted
y = [0, 0, 1, 1, 0, 0, 1, 0, 1, 0]

print("====================================================")
print("Dataset (Age & Income):")
for idx, (features, label) in enumerate(zip(X, y)):
    status = "Accepted" if label == 1 else "Declined"
    print(f" - Cust {idx+1:2d}: Age: {features[0]}, Income: ${features[1]}k -> {status}")

# 2. Standardization Helpers
# Calculate mean for each column
mean_age = sum(p[0] for p in X) / len(X)
mean_inc = sum(p[1] for p in X) / len(X)

# Calculate standard deviation for each column
variance_age = sum((p[0] - mean_age)**2 for p in X) / len(X)
variance_inc = sum((p[1] - mean_inc)**2 for p in X) / len(X)

std_age = math.sqrt(variance_age)
std_inc = math.sqrt(variance_inc)

print(f"\nCalculated Scaling Parameters:")
print(f" - Age   : Mean = {mean_age:.2f}, Std Dev = {std_age:.2f}")
print(f" - Income: Mean = {mean_inc:.2f}, Std Dev = {std_inc:.2f}")

# Function to scale a single coordinate point [Age, Income]
def scale_features(point):
    scaled_age = (point[0] - mean_age) / std_age
    scaled_inc = (point[1] - mean_inc) / std_inc
    return [scaled_age, scaled_inc]

# Standardize the training dataset
X_scaled = [scale_features(pt) for pt in X]

# 3. KNN Prediction Algorithm
def predict_knn(query_point, k=3):
    # Scale query point
    query_scaled = scale_features(query_point)
    
    # Calculate Euclidean distances to all training points
    distances = []
    for idx, train_pt in enumerate(X_scaled):
        dist = math.sqrt((query_scaled[0] - train_pt[0])**2 + (query_scaled[1] - train_pt[1])**2)
        distances.append((dist, idx))
        
    # Sort distances (ascending order)
    distances.sort(key=lambda x: x[0])
    
    # Get top K neighbors
    neighbors = distances[:k]
    
    # Count votes
    votes_accepted = 0
    votes_declined = 0
    print(f"\nNeighbors Selected for voting (K={k}):")
    for dist, idx in neighbors:
        label = y[idx]
        status = "Accepted" if label == 1 else "Declined"
        print(f" - Cust {idx+1}: Age {X[idx][0]}, Income ${X[idx][1]}k | Status: {status} (Distance: {dist:.4f})")
        
        if label == 1:
            votes_accepted += 1
        else:
            votes_declined += 1
            
    # Final prediction
    prediction = 1 if votes_accepted > votes_declined else 0
    return prediction, votes_accepted, votes_declined

# 4. Test prediction on new customer: Age 35, Income 95k
query = [35, 95]
pred, v_acc, v_dec = predict_knn(query, k=3)

final_outcome = "Accepted" if pred == 1 else "Declined"
print(f"\nFinal Prediction for Customer (Age: 35, Income: $95k):")
print(f" - Decision: {final_outcome} (Votes Accepted: {v_acc}, Votes Declined: {v_dec})")
print("====================================================")

"""
M16-A1 - Section B - Task 1
Save & Load a Delivery Time Predictor

Trains a simple LinearRegression model on dummy QuickBite delivery data,
saves it with Joblib, reloads it, and checks the reloaded model gives
the exact same predictions as the original.
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ---------------------------------------------------------
# 1. Generate dummy delivery data
# ---------------------------------------------------------
np.random.seed(42)
N = 200  # more than the required 100 rows

distance_km = np.random.uniform(0.5, 15, N)
num_items = np.random.randint(1, 8, N)
rain_flag = np.random.randint(0, 2, N)

# a simple "true" relationship with some noise, just for dummy data
delivery_time_min = (
    8
    + distance_km * 2.5
    + num_items * 1.2
    + rain_flag * 6
    + np.random.normal(0, 3, N)
)

df = pd.DataFrame({
    "distance_km": distance_km,
    "num_items": num_items,
    "rain_flag": rain_flag,
    "delivery_time_min": delivery_time_min,
})

X = df[["distance_km", "num_items", "rain_flag"]]
y = df["delivery_time_min"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------
# 2. Train the model
# ---------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
print(f"Test RMSE before saving: {rmse:.3f} minutes")

# ---------------------------------------------------------
# 3. Save with Joblib
# ---------------------------------------------------------
MODEL_PATH = "delivery_model.joblib"
joblib.dump(model, MODEL_PATH)

file_size_kb = os.path.getsize(MODEL_PATH) / 1024
print(f"Model saved to '{MODEL_PATH}' ({file_size_kb:.2f} KB)")

# ---------------------------------------------------------
# 4. Reload and verify predictions match
# ---------------------------------------------------------
reloaded_model = joblib.load(MODEL_PATH)

new_sample = pd.DataFrame({
    "distance_km": [4.2],
    "num_items": [3],
    "rain_flag": [1],
})

original_pred = model.predict(new_sample)[0]
reloaded_pred = reloaded_model.predict(new_sample)[0]

print(f"Original model prediction:  {original_pred:.4f}")
print(f"Reloaded model prediction:  {reloaded_pred:.4f}")

if np.isclose(original_pred, reloaded_pred):
    print("PASS: reloaded model output matches original model output")
else:
    print("FAIL: reloaded model output does NOT match original model output")

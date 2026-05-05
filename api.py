from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

app = FastAPI(title="Food Delivery Time Predictor")

# Load our math from the Jupyter notebook!
with open('model_artifacts.pkl', 'rb') as f:
    artifacts = pickle.load(f)

theta = artifacts['theta']
X_mean = artifacts['X_mean']
X_std = artifacts['X_std']
feature_cols = artifacts['feature_cols']

# Define what the incoming JSON should look like
class OrderRequest(BaseModel):
    age: int
    rating: float
    distance_km: float
    vehicle_type: str  # e.g., 'scooter', 'motorcycle'
    order_type: str    # e.g., 'Snack', 'Meal'

@app.post("/predict", response_model=dict)
def predict_time(order: OrderRequest):
    # 1. Calculate distance bucket dynamically
    if order.distance_km>=0 and order.distance_km <= 6: bucket = 1
    elif order.distance_km >=7 and order.distance_km <= 12: bucket = 2
    elif order.distance_km >=13 and order.distance_km <= 20: bucket = 3
    else: bucket = 4
        
    # 2. Standardize Continuous Inputs
    std_age = (order.age - X_mean[0]) / X_std[0]
    std_rating = (order.rating - X_mean[1]) / X_std[1]
    std_dist = (order.distance_km - X_mean[2]) / X_std[2]
    std_bucket = (bucket - X_mean[3]) / X_std[3]
    
    # 3. Handle One-Hot Encoding manually for the API
    # We create a dictionary of all our dummy columns set to 0.0
    # The [4:] slices off the continuous columns to just get the dummies
    dummy_dict = {col: 0.0 for col in feature_cols[4:]}
    
    # If the user's input matches a column we kept, flip it to 1.0!
    expected_vehicle_col = f"Type_of_vehicle_{order.vehicle_type}"
    expected_order_col = f"Type_of_order_{order.order_type}"
    
    if expected_vehicle_col in dummy_dict:
        dummy_dict[expected_vehicle_col] = 1.0
    if expected_order_col in dummy_dict:
        dummy_dict[expected_order_col] = 1.0
        
    # Extract just the 0/1 values in the exact right order
    dummy_values = list(dummy_dict.values())
    
    # 4. Build the final array: [Intercept, Continuous..., Dummies...]
    x_input = np.array([1.0, std_age, std_rating, std_dist, std_bucket] + dummy_values)
    
    # 5. Math!
    prediction = np.dot(x_input, theta)
    
    return {"predicted_time_mins": round(prediction, 2)}

# Run this using the terminal command: uvicorn api:app --reload
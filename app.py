import streamlit as st
import requests

st.title("Delivery Time Prediction")
st.write("Enter the details of your food delivery order to predict the delivery time!")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Driver Age", 18, 65, 30)
    rating = st.slider("Driver Rating", 1.0, 5.0, 4.5, 0.1)
    distance_km = st.slider("Distance (km)", min_value=0.1, max_value=30.0, value=5.0, step=0.1)

with col2:
    vehicle_type = st.selectbox("Vehicle Type", ['scooter', 'motorcycle', 'electric_scooter', 'bicycle'])
    order_type = st.selectbox("Order Type", ['Snack', 'Meal', 'Buffet', 'Beverage'])

if st.button("Predict Delivery Time"):
    order_data = {
        "age": age,
        "rating": rating,
        "distance_km": distance_km,
        "vehicle_type": vehicle_type,
        "order_type": order_type
    }
    
    response = requests.post("http://localhost:8000/predict", json=order_data)
    
    try:
        if response.status_code == 200:
            time = response.json().get("predicted_time_mins", "N/A")
            st.success(f"Predicted Delivery Time: {time} minutes")
        else:
            st.error("Error in prediction. Please try again.")
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed from API: {e}")
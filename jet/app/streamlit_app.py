import streamlit as st
import joblib
import pandas as pd
import os

# Define paths based on project structure
model_path = os.path.join("models", "final_used_car_model(nb).pkl")

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load(model_path)

model = load_model()

# Streamlit UI
st.title("Used Car Price Prediction")
st.markdown("### Enter the details below to predict the car price:")

# Input Fields
year = st.text_input("Year (e.g., 2015)")
kilometers_driven = st.text_input("Kilometers Driven")
mileage = st.text_input("Mileage (km/l)")
engine = st.text_input("Engine (CC)")
power = st.text_input("Power (BHP)")
seats = st.text_input("Seats")

# Categorical Inputs (Example, adjust as needed)
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
owner_type = st.selectbox("Owner Type", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"])

# Submit Button
if st.button("Predict Price"):
    try:
        # Convert inputs to DataFrame
        input_data = pd.DataFrame([{
            "Year": pd.to_numeric(year),
            "Kilometers_Driven": pd.to_numeric(kilometers_driven),
            "Mileage": pd.to_numeric(mileage),
            "Engine": pd.to_numeric(engine),
            "Power": pd.to_numeric(power),
            "Seats": pd.to_numeric(seats),
            "Fuel_Type_" + fuel_type: 1,
            "Transmission_" + transmission: 1,
            "Owner_Type_" + owner_type: 1
        }])

        # Ensure missing categorical values are set to 0
        all_features = model.feature_names_in_
        for col in all_features:
            if col not in input_data.columns:
                input_data[col] = 0

        # Align columns with training data
        input_data = input_data[all_features]

        # Predict
        prediction = model.predict(input_data)

        # Display Result
        st.success(f"**Predicted Car Price: ₹{round(prediction[0], 2)} Lakhs**")

    except Exception as e:
        st.error(f"Error: {str(e)}")


import streamlit as st
import joblib
import pandas as pd
import os

# Adjust path to point to the correct model location
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(base_dir, "models", "final_used_car_model.pkl")

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_model():
    return joblib.load(model_path)

model = load_model()

# Streamlit UI
st.title("Used Car Price Prediction")
st.markdown("### Enter the details below to predict the car price:")

# Function to safely convert inputs
def safe_convert(value):
    try:
        return pd.to_numeric(value)
    except ValueError:
        return None

# Input Fields
year = safe_convert(st.text_input("Year (e.g., 2015)"))
kilometers_driven = safe_convert(st.text_input("Kilometers Driven"))
mileage = safe_convert(st.text_input("Mileage (km/l)"))
engine = safe_convert(st.text_input("Engine (CC)"))
power = safe_convert(st.text_input("Power (BHP)"))
seats = safe_convert(st.text_input("Seats"))

# Categorical Inputs
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
owner_type = st.selectbox("Owner Type", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"])

# Submit Button
if st.button("Predict Price"):
    try:
        if None in [year, kilometers_driven, mileage, engine, power, seats]:
            st.error("Please enter valid numeric values.")
            st.stop()

        # One-hot encode categorical variables
        fuel_type_cols = {f"Fuel_Type_{ft}": 0 for ft in ["Petrol", "Diesel", "CNG", "LPG", "Electric"]}
        transmission_cols = {f"Transmission_{t}": 0 for t in ["Manual", "Automatic"]}
        owner_type_cols = {f"Owner_Type_{ot}": 0 for ot in ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"]}

        fuel_type_cols[f"Fuel_Type_{fuel_type}"] = 1
        transmission_cols[f"Transmission_{transmission}"] = 1
        owner_type_cols[f"Owner_Type_{owner_type}"] = 1

        # Convert inputs to DataFrame
        input_data = pd.DataFrame([{
            "Year": year,
            "Kilometers_Driven": kilometers_driven,
            "Mileage": mileage,
            "Engine": engine,
            "Power": power,
            "Seats": seats,
            **fuel_type_cols,
            **transmission_cols,
            **owner_type_cols
        }])

        # Align with model features
        all_features = getattr(model, "feature_names_in_", None)
        if all_features is None:
            st.error("Model does not have feature names stored. Check the training pipeline.")
            st.stop()

        for col in all_features:
            if col not in input_data.columns:
                input_data[col] = 0

        # Reorder columns
        input_data = input_data[all_features]

        # Predict
        prediction = model.predict(input_data)

        # Display Result
        st.success(f"**Predicted Car Price: ₹{round(prediction[0], 2)} Lakhs**")

    except Exception as e:
        st.error(f"Error: {str(e)}")
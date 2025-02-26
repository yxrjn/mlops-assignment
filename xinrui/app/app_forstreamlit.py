import os
import pandas as pd
import streamlit as st
from pycaret.classification import load_model, predict_model

# Ensure the correct path to the model file
model_path = "final_wheat_seeds_model.pkl"

# Streamlit UI Title
st.title("Wheat Seed Classification App")
st.markdown("Enter the characteristics of the wheat seed to predict its type.")

# Check if the model file exists before loading
if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}. Ensure training was successful.")
    st.stop()

# Load the saved model
model = load_model(model_path.replace(".pkl", ""))
st.success("Model loaded successfully!")

# Define input fields
features = ['Area', 'Perimeter', 'Compactness', 'Length', 'Width', 'AsymmetryCoeff', 'Groove']
input_values = {}

# Collect user inputs
for feature in features:
    input_values[feature] = st.number_input(f"{feature}", value=0.0, step=0.1)

# Predict button
if st.button("Predict Wheat Type"):
    # Create a DataFrame for prediction
    input_df = pd.DataFrame([input_values])

    # Generate prediction using the loaded model
    prediction_df = predict_model(model, data=input_df)

    # Extract numeric prediction and score
    numeric_prediction = prediction_df['prediction_label'].iloc[0] if 'prediction_label' in prediction_df.columns else None
    prediction_score = prediction_df['prediction_score'].iloc[0] if 'prediction_score' in prediction_df.columns else "N/A"

    # Map numeric prediction to wheat type
    wheat_mapping = {1: "Kama", 2: "Rosa", 3: "Canadian"}
    predicted_type = wheat_mapping.get(numeric_prediction, "Unknown")

    # Display prediction results
    st.success(f"Predicted Wheat Type: **{predicted_type}**")
    st.write(f"Prediction Score: **{prediction_score}**")

    # Show input values in a table
    result_df = pd.DataFrame(list(input_values.items()), columns=["Parameter", "Value"])
    result_df.loc[len(result_df)] = ["Predicted Wheat Type", predicted_type]
    result_df.loc[len(result_df)] = ["Prediction Score", prediction_score]

    st.table(result_df)

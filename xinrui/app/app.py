import os
import pandas as pd
import streamlit as st
from flask import Flask, request, render_template
from pycaret.classification import load_model, predict_model
import threading

# Flask App Initialization
app = Flask(__name__)

# Ensure the correct path to the model file
model_path = os.path.join(os.path.dirname(__file__), "final_wheat_seeds_model.pkl")

# Debug print to verify the path
print(f"Loading model from: {model_path}")

# Check if the model file exists before loading
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}. Ensure training was successful.")

# Load the saved model
model = load_model(model_path.replace(".pkl", ""))

print("Model loaded successfully!")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Define the feature names as used in training
        features = ['Area', 'Perimeter', 'Compactness', 'Length', 'Width', 'AsymmetryCoeff', 'Groove']

        # Extract input values from the form and convert them to float
        input_values = [float(request.form.get(feature)) for feature in features]

        # Create a DataFrame for prediction
        input_df = pd.DataFrame([input_values], columns=features)

        # Generate prediction using the loaded model
        prediction_df = predict_model(model, data=input_df)

        # Extract numeric prediction and score
        numeric_prediction = prediction_df['prediction_label'].iloc[0] if 'prediction_label' in prediction_df.columns else None
        prediction_score = prediction_df['prediction_score'].iloc[0] if 'prediction_score' in prediction_df.columns else "N/A"

        # Map numeric prediction to wheat type
        wheat_mapping = {1: "Kama", 2: "Rosa", 3: "Canadian"}
        predicted_type = wheat_mapping.get(numeric_prediction, "Unknown")

        # Prepare a table (list of dictionaries) to display results
        prediction_table = [
            {"Parameter": "Area", "Value": input_df["Area"].iloc[0]},
            {"Parameter": "Perimeter", "Value": input_df["Perimeter"].iloc[0]},
            {"Parameter": "Compactness", "Value": input_df["Compactness"].iloc[0]},
            {"Parameter": "Length", "Value": input_df["Length"].iloc[0]},
            {"Parameter": "Width", "Value": input_df["Width"].iloc[0]},
            {"Parameter": "AsymmetryCoeff", "Value": input_df["AsymmetryCoeff"].iloc[0]},
            {"Parameter": "Groove", "Value": input_df["Groove"].iloc[0]},
            {"Parameter": "Predicted Wheat Type", "Value": predicted_type},
            {"Parameter": "Prediction Score", "Value": prediction_score},
        ]

        return render_template('index.html', prediction_table=prediction_table, prediction_text=f"Predicted Wheat Type: {predicted_type}")

    except Exception as e:
        # In case of an error, display the error message on the page
        return render_template('index.html', prediction_text=f'Error: {str(e)}')


# 🌟 Streamlit UI for Alternative Access
def run_streamlit():
    st.title("🌾 Wheat Seed Classification App")
    st.markdown("Enter the characteristics of the wheat seed to predict its type.")

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


if __name__ == '__main__':
    # Run Flask and Streamlit in parallel
    flask_thread = threading.Thread(target=lambda: app.run(debug=True, use_reloader=False))
    flask_thread.start()

    run_streamlit()

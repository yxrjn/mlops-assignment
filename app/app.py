from flask import Flask, request, render_template
import pandas as pd
import pickle
import os
from pycaret.classification import load_model, predict_model

app = Flask(__name__)

# Get the base directory (one level up from `app`)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Load models with correct paths
models = {
    "predict_xr": load_model(os.path.join(BASE_DIR, "xinrui/models/final_wheat_seeds_model")),  # PyCaret Model
    # "predict_jet": pickle.load(open(os.path.join(BASE_DIR, "jet/models/final_used_car_model.pkl"), "rb")),  # Pickle Model
    # "predict_dk": pickle.load(open(os.path.join(BASE_DIR, "dekai/models/final_melbourne_model.pkl"), "rb"))  # Pickle Model
}

# Define feature sets for each model
feature_sets = {
    "predict_xr": ['Area', 'Perimeter', 'Compactness', 'Length', 'Width', 'AsymmetryCoeff', 'Groove'],
    # "predict_jet": ['Kilometers_Driven', 'Fuel_Type', 'Transmission', 'Mileage'],
    # "predict_dk": ['Distance_CBD', 'Land_Size', 'Bedrooms', 'Building_Area', 'Age']
}

# HTML template mapping
model_html_templates = {
    "predict_xr": "xinrui.html",
    # "predict_jet": "jet.html",
    # "predict_dk": "dekai.html"
}

# Mapping for Wheat Classification
wheat_mapping = {1: "Kama", 2: "Rosa", 3: "Canadian"}

@app.route("/")
def home():
    return render_template("index.html", models=list(models.keys()))  # Main selection page

@app.route("/predict", methods=["POST"])
def predict():
    try:
        model_type = request.form.get("model", "predict_xr")  # Default to predict_xr if no model selected
        if model_type not in models:
            return render_template("index.html", prediction_text="Error: Invalid model selection", models=list(models.keys()))

        # Extract feature values dynamically
        features = feature_sets[model_type]
        input_values = [request.form.get(feature) for feature in features]

        # Convert inputs to float where applicable
        try:
            input_values = [float(value) for value in input_values]
        except ValueError:
            return render_template(model_html_templates[model_type], prediction_text="Error: Invalid input format. Ensure all values are numbers.")

        input_df = pd.DataFrame([input_values], columns=features)

        if model_type == "predict_xr":  # For xinrui's model (Wheat Seeds Classification)
            prediction_df = predict_model(models[model_type], data=input_df)

            numeric_prediction = prediction_df['prediction_label'].iloc[0] if 'prediction_label' in prediction_df.columns else None
            prediction_score = prediction_df['prediction_score'].iloc[0] if 'prediction_score' in prediction_df.columns else "N/A"

            predicted_type = wheat_mapping.get(numeric_prediction, "Unknown")

            prediction_table = [{"Parameter": feature, "Value": input_df[feature].iloc[0]} for feature in features]
            prediction_table.append({"Parameter": "Predicted Wheat Type", "Value": predicted_type})
            prediction_table.append({"Parameter": "Prediction Score", "Value": prediction_score})

            return render_template("xinrui.html", prediction_table=prediction_table)

        else:  # Pickle Models (Used Car / Melbourne Housing)
            prediction = models[model_type].predict(input_df)[0]
            return render_template(model_html_templates[model_type], prediction_text=f"Predicted Result: {prediction}")

    except Exception as e:
        return render_template(model_html_templates.get(model_type, "index.html"), prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True, host="192.168.0.25", port=5000)


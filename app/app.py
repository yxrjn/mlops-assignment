from flask import Flask, request, render_template
import pandas as pd
import pickle
import os
from pycaret.classification import load_model, predict_model

app = Flask(__name__)

# Get the base directory (one level up from app)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Load models with correct paths
models = {
    "predict_xr": load_model(os.path.join(BASE_DIR, "xinrui/app/final_wheat_seeds_model")),  # PyCaret Model
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
# Mapping model names to display names for dropdown
model_mapping = {
    "predict_xr": "Wheat Seeds Prediction",
    "predict_jet": "Used Car Price Prediction",
    "predict_dk": "Melbourne"
}

@app.route('/')
def home():
    default_model = "predict_xr"
    model_display_name = model_mapping.get(default_model, "Unknown Model")

    return render_template('index.html',
                           model_display_name=model_display_name,
                           model_mapping=model_mapping,
                           selected_model=default_model)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        model_type = request.form.get("model", "predict_xr")  # Default to predict_xr if no model selected

        if model_type not in models:
            return render_template("index.html",
                                   prediction_text="Error: Invalid model selection",
                                   model_mapping=model_mapping,
                                   selected_model=model_type)

        features = feature_sets[model_type]
        input_values = [request.form.get(feature) for feature in features]

        try:
            input_values = [float(value) for value in input_values]
        except ValueError:
            return render_template(model_html_templates[model_type],
                                   prediction_text="Error: Invalid input format. Ensure all values are numbers.",
                                   model_mapping=model_mapping,
                                   selected_model=model_type)

        input_df = pd.DataFrame([input_values], columns=features)

        if model_type == "predict_xr":  # For Wheat Seeds Classification
            prediction_df = predict_model(models[model_type], data=input_df)

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
            return render_template("xinrui.html",
                                   prediction_table=prediction_table,
                                   model_mapping=model_mapping,
                                   selected_model=model_type,prediction_text=f"Predicted Wheat Type: {predicted_type}")

        else:  # Pickle Models (Used Car / Melbourne Housing)
            prediction = models[model_type].predict(input_df)[0]
            return render_template(model_html_templates[model_type],
                                   prediction_text=f"Predicted Result: {prediction}",
                                   model_mapping=model_mapping,
                                   selected_model=model_type)

    except Exception as e:
        return render_template(model_html_templates.get(model_type, "index.html"),
                               prediction_text=f"Error: {str(e)}",
                               model_mapping=model_mapping,
                               selected_model=model_type)

if __name__ == "__main__":
    app.run(debug=True, host="192.168.0.25", port=5000)
from flask import Flask, request, render_template
import pandas as pd
import pickle
import os
from pycaret.classification import load_model, predict_model

app = Flask(__name__)

# Get the base directory (one level up from app)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Load models
models = {
    "predict_xr": load_model(os.path.join(BASE_DIR, "xinrui/models/final_wheat_seeds_model")),
    "predict_dk": load_model(os.path.join(BASE_DIR, "dekai/models/final_melbourne_model"))
}

# Define feature sets for each model
feature_sets = {
    "predict_xr": ['Area', 'Perimeter', 'Compactness', 'Length', 'Width', 'AsymmetryCoeff', 'Groove'],
    "predict_dk": ["Rooms", "Type", "Distance", "Bathroom", "Car", "Landsize", "CouncilArea", "Region", "Age"]
}

# HTML template mapping
model_html_templates = {
    "predict_xr": "xinrui.html",
    "predict_dk": "dekais_app.html"
}

# Mapping model names to display names
model_mapping = {
    "predict_xr": "Wheat Seeds Prediction",
    "predict_dk": "Melbourne House Price Prediction"
}

@app.route('/')
def home():
    return render_template('index.html', model_mapping=model_mapping)

@app.route('/select_model', methods=['POST'])
def select_model():
    selected_model = request.form.get("model")
    if selected_model in model_html_templates:
        return render_template(model_html_templates[selected_model], model_mapping=model_mapping)
    return render_template("index.html", model_mapping=model_mapping)

@app.route("/xinrui")
def predict_xr():
    return render_template("xinrui.html", model_mapping=model_mapping)

@app.route("/dekais_app")
def predict_dk():
    return render_template("dekais_app.html", model_mapping=model_mapping)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        model_type = request.form.get("model")  # Get selected model

        if model_type not in models:
            return render_template("index.html", prediction_text="Error: Invalid model selection",
                                   model_mapping=model_mapping)

        features = feature_sets[model_type]
        input_values = [request.form.get(feature) for feature in features]

        if model_type == "predict_xr":  # Wheat Seeds Prediction
            try:
                input_values = [float(value) for value in input_values]
            except ValueError:
                return render_template("xinrui.html", prediction_text="Error: Ensure all values are numbers.",
                                       model_mapping=model_mapping)

            input_df = pd.DataFrame([input_values], columns=features)
            prediction_df = predict_model(models[model_type], data=input_df)

            numeric_prediction = prediction_df['prediction_label'].iloc[
                0] if 'prediction_label' in prediction_df.columns else None
            prediction_score = prediction_df['prediction_score'].iloc[
                0] if 'prediction_score' in prediction_df.columns else "N/A"

            wheat_mapping = {1: "Kama", 2: "Rosa", 3: "Canadian"}
            predicted_type = wheat_mapping.get(numeric_prediction, "Unknown")

            prediction_table = [{"Parameter": feature, "Value": input_df[feature].iloc[0]} for feature in features]
            prediction_table.append({"Parameter": "Predicted Wheat Type", "Value": predicted_type})
            prediction_table.append({"Parameter": "Prediction Score", "Value": prediction_score})

            return render_template("xinrui.html", prediction_table=prediction_table, model_mapping=model_mapping,
                                   prediction_text=f"Predicted Wheat Type: {predicted_type}")

        elif model_type == 'predict_dk':  # Melbourne House Price Prediction
            try:
                # Convert form data into dictionary with correct types
                form_data = {feature: request.form[feature] for feature in features}
                form_data['Rooms'] = int(form_data['Rooms'])
                form_data['Distance'] = float(form_data['Distance'])
                form_data['Bathroom'] = int(form_data['Bathroom'])
                form_data['Car'] = int(form_data['Car'])
                form_data['Landsize'] = float(form_data['Landsize'])
                form_data['Age'] = int(form_data['Age'])

                input_df = pd.DataFrame([form_data])

                # Predict using the model
                dekai_prediction = predict_model(models[model_type], data=input_df)

                # Extract the predicted price
                predicted_price = round(dekai_prediction["prediction_label"].iloc[0], 2)

                return render_template("dekais_app.html", predicted_price=predicted_price, model_mapping=model_mapping)

            except Exception as e:
                return render_template("dekais_app.html", prediction_text=f"Error: {str(e)}",
                                       model_mapping=model_mapping)

        return render_template("index.html", prediction_text="Error: Unknown model type", model_mapping=model_mapping)

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}", model_mapping=model_mapping)


if __name__ == "__main__":
    app.run(debug=True)

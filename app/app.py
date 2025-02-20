from flask import Flask, request, render_template, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load all trained models
models = {
    "predict_xr": pickle.load(open("xr/app/final_wheat_seeds_model.pkl", "rb")),
    "predict_jet": pickle.load(open("jet/app/final_used_car_model.pkl", "rb")),
    "predict_dk": pickle.load(open("dk/app/final_melbourne_model.pkl", "rb"))
}

# Define feature sets for each model
feature_sets = {
    "predict_xr": ['Area', 'Perimeter', 'Compactness', 'Length', 'Width', 'AsymmetryCoeff', 'Groove'],
    "predict_jet": ['Kilometers_Driven', 'Fuel_Type', 'Transmission', 'Mileage'],
    "predict_dk": ['Distance_CBD', 'Land_Size', 'Bedrooms', 'Building_Area', 'Age']
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        model_type = request.form.get("model")
        if model_type not in models:
            return render_template("index.html", prediction_text="Error: Invalid model selection")

        # Extract feature values
        features = feature_sets[model_type]
        input_values = [float(request.form.get(feature)) for feature in features]

        # Create a DataFrame with input data
        input_df = pd.DataFrame([input_values], columns=features)

        # Generate predictions
        prediction = models[model_type].predict(input_df)[0]

        return render_template("index.html", prediction_text=f"Predicted Result: {prediction}")
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

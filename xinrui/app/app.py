from flask import Flask, request, render_template
import pandas as pd
from pycaret.classification import load_model, predict_model

app = Flask(__name__)

# Load the saved model from Task 2
model = load_model('model/final_wheat_seeds_model')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Define the feature names as used in the model
        features = ['Area', 'Perimeter', 'Compactness', 'Length', 'Width', 'AsymmetryCoeff', 'Groove']

        # Extract feature values from the form; convert them to float
        input_values = [float(request.form.get(feature)) for feature in features]

        # Create a DataFrame from the input values (one row)
        input_df = pd.DataFrame([input_values], columns=features)

        # Generate prediction using the loaded model
        prediction_df = predict_model(model, data=input_df)

        # The predicted label is in the 'Label' column
        prediction = prediction_df['Label'][0]

        return render_template('index.html', prediction_text=f'Predicted Wheat Type: {prediction}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')


if __name__ == '__main__':
    app.run(debug=True)

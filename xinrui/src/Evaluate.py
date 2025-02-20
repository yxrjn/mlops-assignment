import os
import pandas as pd
import hydra
from omegaconf import DictConfig
from pycaret.classification import load_model, predict_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json

@hydra.main(config_path=".", config_name="config")
def evaluate_model(cfg: DictConfig):
    model_path = "xr/app/final_wheat_seeds_model.pkl"
    test_data_path = cfg.dataset.processed_test_path
    metrics_path = "xr/metrics.json"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}")
    
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test dataset not found at {test_data_path}")
    
    # Load the model
    model = load_model(model_path)
    
    # Load test data
    test_df = pd.read_csv(test_data_path)
    target = "Type"
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]
    
    # Generate predictions
    predictions = predict_model(model, data=X_test)
    y_pred = predictions["Label"]
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()
    
    # Save metrics
    metrics = {
        "accuracy": accuracy,
        "classification_report": class_report,
        "confusion_matrix": conf_matrix
    }
    
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Evaluation complete. Metrics saved to {metrics_path}")

if __name__ == "__main__":
    evaluate_model()

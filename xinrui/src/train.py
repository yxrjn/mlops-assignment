import os
import pandas as pd
import hydra
import matplotlib
import shutil
matplotlib.use("Agg") 
from omegaconf import DictConfig
from pycaret.classification import (
    setup, compare_models, evaluate_model, plot_model, 
    predict_model, create_model, save_model
)

@hydra.main(config_path=".", config_name="config", version_base=None)
def train_model(cfg: DictConfig):
    # Manually set the working directory to the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    os.chdir(project_root)
    dataset_path = os.path.abspath(cfg.dataset.processed_train_path)
    print(f"Checking dataset path: {dataset_path}")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Processed dataset file '{dataset_path}' not found. Run preprocess.py first.")

    # Load dataset
    df = pd.read_csv(dataset_path)

    # Set up PyCaret
    clf_setup = setup(
        data=df,
        target="Type",
        normalize=cfg.training.normalize,
        feature_selection=cfg.training.feature_selection,
        bin_numeric_features=cfg.training.get("bin_features", []),  # Avoid error if missing
        session_id=cfg.training.session_id,
        log_experiment=cfg.training.get("log_experiment", False),
    )

    # Automatically select best model or use predefined one
    if cfg.model.auto_select:
        print("Auto-selecting best model...")
        best_model = compare_models(sort="Accuracy")
    else:
        if not cfg.model.type:
            raise ValueError("Model type must be specified in config.yaml if auto_select is False.")
        print(f"Creating model: {cfg.model.type}")
        best_model = create_model(cfg.model.type)

    # Evaluate the model
    print("Evaluating model performance...")
    evaluate_model(best_model)
    
    save_dir = "xinrui/plot"

    os.makedirs(save_dir, exist_ok=True)

    plot_model(best_model, plot="confusion_matrix", save=True)
    plot_model(best_model, plot="auc", save=True)

   
    shutil.move("Confusion Matrix.png", f"{save_dir}/confusion_matrix.png")
    shutil.move("AUC.png", f"{save_dir}/auc.png")

    print(f"Confusion Matrix saved to: {save_dir}/confusion_matrix.png")
    print(f"AUC Curve saved to: {save_dir}/auc.png")
    if hasattr(best_model, "coef_") or hasattr(best_model, "feature_importances_"):
        plot_model(best_model, plot="feature")
    else:
        print("Feature Importance plot skipped as model does not support it.")

    # Generate predictions
    print("Making predictions on test data...")
    predictions = predict_model(best_model)

    # Save the trained model
    model_path = os.path.abspath(cfg.model.output_path)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)  # Ensure directory exists
    save_model(best_model, model_path)
    print(f"Model saved to {model_path}")

    return predictions

if __name__ == "__main__":
    train_model()

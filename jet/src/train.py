import os
import pandas as pd
import hydra
import logging
import shutil
from omegaconf import DictConfig
from pycaret.regression import setup, compare_models, evaluate_model, plot_model, create_model, save_model
import mlflow

# Ensure MLflow tracking is set up
mlflow.set_tracking_uri("http://localhost:5000")  # Ensure MLflow server is running
print("MLflow tracking URI set successfully!")

# Initialize logging
log_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(log_dir, "training_logs.txt")

logging.basicConfig(
    filename=log_file,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

@hydra.main(config_path=".", config_name="config", version_base=None)
def train_model(cfg: DictConfig):
    # Get the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))  # Move one level up

    # Define dataset path
    dataset_path = os.path.abspath(os.path.join(script_dir, cfg.dataset.processed_train_path))

    logging.info(f"Checking dataset path: {dataset_path}")

    if not os.path.exists(dataset_path):
        logging.error(f"Processed dataset file '{dataset_path}' not found. Run preprocess.py first.")
        raise FileNotFoundError(f"Processed dataset file '{dataset_path}' not found. Run preprocess.py first.")

    # Load dataset
    df = pd.read_csv(dataset_path)
    # Ensure train.py reads column names in Title Case
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("-", "_").str.title()

    # Drop duplicate columns before passing to PyCaret
    duplicate_columns = df.columns[df.columns.duplicated()]
    if not duplicate_columns.empty:
        print(f"Dropping duplicate columns: {duplicate_columns.tolist()}")
        df = df.loc[:, ~df.columns.duplicated()]

    # Set up PyCaret
    clf_setup = setup(
        data=df,
        target="Price_(Inr_Lakhs)",
        normalize=cfg.training.normalize,
        feature_selection=cfg.training.feature_selection,
        bin_numeric_features=cfg.training.get("bin_features", []),
        session_id=cfg.training.session_id,
        log_experiment=cfg.training.get("log_experiment", True),
    )

    # Auto-select best model or use a predefined model
    if cfg.model.auto_select:
        logging.info("Auto-selecting best model...")
        best_model = compare_models(sort="R2")
    else:
        if not cfg.model.type:
            logging.error("Model type must be specified in config.yaml if auto_select is False.")
            raise ValueError("Model type must be specified in config.yaml if auto_select is False.")

        logging.info(f"Creating model: {cfg.model.type}")
        best_model = create_model(cfg.model.type)

    # Evaluate the model
    logging.info("Evaluating model performance...")
    evaluate_model(best_model)

    # Save plots
    save_dir = os.path.abspath(os.path.join(script_dir, "..", "plot"))
    os.makedirs(save_dir, exist_ok=True)

    plot_model(best_model, plot="residuals", save=True)
    plot_model(best_model, plot="error", save=True)

    shutil.move("Residuals.png", os.path.join(save_dir, "residuals.png"))
    shutil.move("Prediction Error.png", os.path.join(save_dir, "error.png"))

    logging.info("Residual and error plots saved.")

    # Save the trained model
    model_path = os.path.abspath(os.path.join(script_dir, cfg.model.output_path))
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    save_model(best_model, model_path)

    logging.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    logging.info("Starting training script...")
    train_model()
    logging.info("Training completed successfully.")

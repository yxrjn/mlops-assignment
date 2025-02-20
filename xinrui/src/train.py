import os
import pandas as pd
import hydra
from omegaconf import DictConfig
from pycaret.classification import setup, compare_models, evaluate_model, plot_model, predict_model, create_model

@hydra.main(config_path=".", config_name="config")
def train_model(cfg: DictConfig):
    dataset_path = cfg.dataset.raw_path
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file '{dataset_path}' not found.")

    df = pd.read_csv(dataset_path)

    clf_setup = setup(
        data=df,
        target="Type",
        normalize=cfg.training.normalize,
        feature_selection=cfg.training.feature_selection,
        bin_numeric_features=cfg.training.bin_features,
        session_id=cfg.training.session_id,
        log_experiment=cfg.training.log_experiment,
    )

    # Automatically select best model or use predefined one
    if cfg.model.auto_select:
        best_model = compare_models(sort="Accuracy")
    else:
        best_model = create_model(cfg.model.type)

    evaluate_model(best_model)
    plot_model(best_model, plot="confusion_matrix")
    plot_model(best_model, plot="auc")

    if hasattr(best_model, "coef_") or hasattr(best_model, "feature_importances_"):
        plot_model(best_model, plot="feature")
    else:
        print("Feature Importance plot skipped as model does not support it.")

    predictions = predict_model(best_model)
    return predictions

if __name__ == "__main__":
    train_model()

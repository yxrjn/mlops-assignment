import os
import pandas as pd
import hydra
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

@hydra.main(config_path=".", config_name="config")
def preprocess_data(cfg: DictConfig):
    dataset_path = cfg.dataset.raw_path
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file '{dataset_path}' not found.")

    # Load dataset
    df = pd.read_csv(dataset_path)

    #features and target variable
    features = ["Area", "Perimeter", "Compactness", "Length", "Width", "AsymmetryCoeff", "Groove"]
    target = "Type"

    # Standardize numerical features
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    # Split into train and test sets
    train_df, test_df = train_test_split(df, test_size=cfg.dataset.test_size, random_state=cfg.training.session_id, stratify=df[target])

    # Save processed data
    train_df.to_csv(cfg.dataset.processed_train_path, index=False)
    test_df.to_csv(cfg.dataset.processed_test_path, index=False)

    print(f"Preprocessing complete. Train set saved to {cfg.dataset.processed_train_path}, Test set saved to {cfg.dataset.processed_test_path}")

if __name__ == "__main__":
    preprocess_data()

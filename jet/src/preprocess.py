import os
import pandas as pd
import hydra
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split


@hydra.main(config_path=".", config_name="config", version_base=None)
def preprocess_data(cfg: DictConfig):
    # Get absolute dataset path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.abspath(os.path.join(script_dir, cfg.dataset.raw_path))

    print(f"Checking dataset path: {dataset_path}")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file '{dataset_path}' not found.")

    # Read dataset (Detect format)
    if dataset_path.endswith(".csv"):
        df = pd.read_csv(dataset_path, encoding="utf-8")
    elif dataset_path.endswith(".xlsx"):
        df = pd.read_excel(dataset_path, engine="openpyxl")
    else:
        raise ValueError("Unsupported file format! Use CSV or XLSX.")

    # **Fix column names (lowercase, underscores)**
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("-", "_").str.lower()

    # Extract feature lists
    numeric_features = [col.lower() for col in cfg.dataset.numeric_features]
    categorical_features = [col.lower() for col in cfg.dataset.categorical_features]

    # **Validate numeric columns exist**
    missing_numeric = [col for col in numeric_features if col not in df.columns]
    if missing_numeric:
        raise KeyError(f"Missing numeric columns: {missing_numeric}")

    # **Validate categorical columns exist**
    missing_categorical = [col for col in categorical_features if col not in df.columns]
    categorical_features = [col for col in categorical_features if col in df.columns]  # Keep only existing ones

    # **Ensure numeric columns are properly formatted**
    for col in numeric_features:
        df[col] = df[col].astype(str).str.extract("([-+]?\d*\.?\d+)")[0]
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col].fillna(df[col].median(), inplace=True)

    # One-hot encode categorical features and check for duplicates
    if categorical_features:
        df = pd.get_dummies(df, columns=categorical_features, drop_first=True)
        
        # Verify if duplicates arise due to one-hot encoding
        duplicate_columns_after_encoding = df.columns[df.columns.duplicated()]
        if not duplicate_columns_after_encoding.empty:
            print(f"⚠ Duplicates detected AFTER one-hot encoding: {duplicate_columns_after_encoding.tolist()}")

        
    # Identify duplicate columns properly
    duplicate_columns = df.columns[df.columns.duplicated(keep=False)]

    if not duplicate_columns.empty:
        print(f"❌ Duplicate columns found: {duplicate_columns.tolist()}")

    # Drop duplicate columns correctly
    df = df.loc[:, ~df.columns.duplicated(keep="first")]


    # **Split into train and test sets**
    train_df, test_df = train_test_split(
        df, test_size=cfg.dataset.test_size, random_state=cfg.training.session_id
    )
    # Ensure column names match PyCaret's format before saving
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("-", "_").str.title()

    # **Save processed data**
    processed_train_path = os.path.abspath(os.path.join(script_dir, cfg.dataset.processed_train_path))
    processed_test_path = os.path.abspath(os.path.join(script_dir, cfg.dataset.processed_test_path))

    os.makedirs(os.path.dirname(processed_train_path), exist_ok=True)
    os.makedirs(os.path.dirname(processed_test_path), exist_ok=True)

    train_df.to_csv(processed_train_path, index=False)
    test_df.to_csv(processed_test_path, index=False)

    print(f"Preprocessing complete. Train set saved to {processed_train_path}, Test set saved to {processed_test_path}")


if __name__ == "__main__":
    preprocess_data()

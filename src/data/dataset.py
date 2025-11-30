import pandas as pd
import yaml

def load_config():
    with open("src/config/config.yaml", "r") as f:
        return yaml.safe_load(f)

def load_dataset(split="train", feature_set="baseline"):
    config = load_config()
    base_path = config["paths"]["data_raw"]

    if split == "train":
        X = pd.read_csv(f"{base_path}/X_train.csv", low_memory=False)
        y = pd.read_csv(f"{base_path}/y_train.csv").values.ravel()
    else:
        X = pd.read_csv(f"{base_path}/X_test.csv", low_memory=False)
        y = None

    features = config["features"][feature_set]
    X = X[features].copy()

    # Nettoyage minimal
    X["possibly_sensitive"] = X["possibly_sensitive"].fillna(0).astype(int)
    X["user.description"] = X["user.description"].fillna("")
    X["is_quote_status"] = X["is_quote_status"].fillna(0).astype(int)

    return X, y

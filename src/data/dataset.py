# import pandas as pd
# import yaml

# def load_config():
#     with open("src/config/config.yaml", "r") as f:
#         return yaml.safe_load(f)

# def load_dataset(split="train", feature_set="baseline"):
#     config = load_config()
#     base_path = config["paths"]["data_raw"]

#     if split == "train":
#         X = pd.read_csv(f"{base_path}/X_train.csv", low_memory=False)
#         y = pd.read_csv(f"{base_path}/y_train.csv").values.ravel()
#     else:
#         X = pd.read_csv(f"{base_path}/X_test.csv", low_memory=False)
#         y = None

#     features = config["features"][feature_set]
#     X = X[features].copy()

#     # Nettoyage minimal
#     X["possibly_sensitive"] = X["possibly_sensitive"].fillna(0).astype(int)
#     X["user.description"] = X["user.description"].fillna("")
#     X["is_quote_status"] = X["is_quote_status"].fillna(0).astype(int)

#     return X, y

import pandas as pd

def load_dataset(split="train", feature_set="baseline", version="essential"):
    if version == "essential":
            if split == "train":
                X = pd.read_csv("data/processed/X_train_essential.csv")
                y = pd.read_csv("data/processed/y_train_essential.csv").iloc[:, 0]
            else:
                X = pd.read_csv("data/processed/X_kaggle_essential.csv")
                y = None
    elif version == "full":
            if split == "train":
                X = pd.read_csv("data/raw/X_train.csv")
                y = pd.read_csv("data/raw/y_train.csv").iloc[:, 0]
            else:
                X = pd.read_csv("data/raw/X_kaggle.csv")
                y = None
    else:   
        raise ValueError("Version du dataset non reconnue.")

    return X, y

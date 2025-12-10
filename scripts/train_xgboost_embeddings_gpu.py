"""
Entraînement XGBoost GPU sur embeddings avec :
- API native xgb.train (loss + early stopping OK)
- tqdm
- logging JSON
- sauvegarde du meilleur modèle
"""

import os
import json
import time
import numpy as np
import xgboost as xgb

from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split

from src.data.dataset import load_dataset
from src.utils.seed import set_seed
from src.utils.model_saving import save_model

# ============================================================
# ✅ PARAMÈTRES À TESTER (VERSION ACTUELLE)
# ============================================================

param_list = [
    {
        "n_estimators": 800,
        "max_depth": 9,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 5,
    }
]

# ============================================================
# ✅ CHEMINS
# ============================================================

LOG_DIR = "logs"
MODEL_DIR = "models"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# ✅ MAIN
# ============================================================

def main():
    set_seed(42)

    print("Chargement du dataset...")
    X, y = load_dataset(split="train", version="embeddings")
    
    n = min(len(X), len(y))
    X = X.iloc[:n].reset_index(drop=True)
    y = y.iloc[:n].reset_index(drop=True)

    # Sécurité : conversion numpy
    if not isinstance(X, np.ndarray):
        X = X.values

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    non_numeric_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    if len(non_numeric_cols) > 0:
        print("⚠️ Colonnes non numériques encore présentes :")
        for c in non_numeric_cols:
            print(" -", c)
        raise ValueError("Colonnes non numériques détectées dans X_train")
    
    X_train = X_train.select_dtypes(include=[np.number])
    X_val   = X_val.select_dtypes(include=[np.number])



    # Sélection stricte des colonnes numériques uniquement
    X_train_num = X_train.select_dtypes(include=[np.number]).to_numpy()
    X_val_num   = X_val.select_dtypes(include=[np.number]).to_numpy()

    y_train_np = y_train.to_numpy()
    y_val_np   = y_val.to_numpy()

    dtrain = xgb.DMatrix(X_train_num, label=y_train_np)
    dval   = xgb.DMatrix(X_val_num,   label=y_val_np)

    best_score = -1
    best_model = None
    best_params = None

    all_logs = []

    # ============================================================
    # ✅ BOUCLE D’ENTRAÎNEMENT
    # ============================================================

    for i, params in enumerate(tqdm(param_list, desc="Grid manuel")):
        print("\n" + "=" * 80)
        print(f"Modèle {i + 1}/{len(param_list)}")
        print(params)

        xgb_params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": params["max_depth"],
            "eta": params["learning_rate"],
            "subsample": params["subsample"],
            "colsample_bytree": params["colsample_bytree"],
            "min_child_weight": params["min_child_weight"],
            "tree_method": "gpu_hist",
            "predictor": "gpu_predictor",
        }

        evals_result = {}

        print("Entraînement GPU...")

        start_time = time.time()

        booster = xgb.train(
            params=xgb_params,
            dtrain=dtrain,
            num_boost_round=params["n_estimators"],
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=50,
            evals_result=evals_result,
            verbose_eval=50,
        )

        train_time = time.time() - start_time
        print(f"⏱️ Temps : {train_time:.1f}s")

        # ============================================================
        # ✅ PRÉDICTIONS & SCORES
        # ============================================================

        y_train_pred = (booster.predict(dtrain) > 0.5).astype(int)
        y_val_pred   = (booster.predict(dval)   > 0.5).astype(int)

        train_f1 = f1_score(y_train, y_train_pred)
        val_f1   = f1_score(y_val,   y_val_pred)

        print(f"Train F1 = {train_f1:.4f}")
        print(f"Val   F1 = {val_f1:.4f}")

        print(classification_report(y_val, y_val_pred))

        # ============================================================
        # ✅ LOG JSON
        # ============================================================

        log_entry = {
            "params": params,
            "train_f1": float(train_f1),
            "val_f1": float(val_f1),
            "n_rounds_used": len(evals_result["train"]["logloss"]),
            "train_loss": evals_result["train"]["logloss"],
            "val_loss": evals_result["val"]["logloss"],
            "train_time_sec": train_time,
        }

        all_logs.append(log_entry)

        with open(os.path.join(LOG_DIR, "xgboost_embeddings_logs.json"), "w") as f:
            json.dump(all_logs, f, indent=2)

        # ============================================================
        # ✅ MEILLEUR MODÈLE
        # ============================================================

        if val_f1 > best_score:
            print("✅ Nouveau meilleur modèle")
            best_score = val_f1
            best_model = booster
            best_params = params

    # ============================================================
    # ✅ SAUVEGARDE DU MEILLEUR MODÈLE
    # ============================================================

    model_path = os.path.join(
        MODEL_DIR,
        f"xgboost_embeddings_best_f1_{best_score:.4f}.json"
    )

    best_model.save_model(model_path)

    print("\n" + "=" * 80)
    print("✅ Entraînement terminé")
    print("Meilleurs paramètres :", best_params)
    print(f"Best F1 = {best_score:.4f}")
    print("Modèle sauvegardé dans :", model_path)


if __name__ == "__main__":
    main()

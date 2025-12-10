"""
Script d'entraînement pour XGBoost avec TF-IDF + SVD + tuning hyperparamètres.
Usage:
    python -m scripts.train_xgboost2
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import classification_report, f1_score

from src.data.dataset import load_dataset
from src.data.preprocessing import add_account_age, add_hashtag_count
from src.models.xgboost2 import build_xgboost_model
from src.utils.seed import set_seed
from src.utils.model_saving import save_model


def main():
    set_seed(42)

    print("Chargement des données...")
    # --- 1) Chargement
    X, y = load_dataset(split="train", version="essential")

    # --- 2) Feature engineering
    X = add_account_age(X)
    X = add_hashtag_count(X)

    # --- 3) Gestion minimale des valeurs manquantes
    for col in ["possibly_sensitive", "is_quote_status"]:
        if col in X.columns:
            X[col] = X[col].fillna(0).astype(int)

    for col in ["user.description", "full_text"]:
        if col in X.columns:
            X[col] = X[col].fillna("").astype(str)

    # --- 4) Split train / validation (pour évaluation finale)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    print("Construction du modèle XGBoost (TF-IDF + SVD)...")
    # --- 5) Build du modèle SVD + XGBoost
    base_model = build_xgboost_model(
        n_components_text=300,
        n_components_desc=100,
    )

    # --- 6) Espace de recherche d'hyperparamètres (adapté au SVD)
    param_dist = {
        "clf__n_estimators": [600, 800, 1000],
        "clf__max_depth": [5, 7, 9],
        "clf__learning_rate": [0.03, 0.05, 0.08],
        "clf__subsample": [0.7, 0.85, 1.0],
        "clf__colsample_bytree": [0.7, 0.85, 1.0],
        "clf__min_child_weight": [1, 3, 5],
    }

    # --- 7) CV stratifiée
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=20,
        scoring="f1",
        cv=cv,
        verbose=2,
        n_jobs=1,        # 1 conseillé avec XGB + GPU
        random_state=42,
    )

    print("Démarrage du tuning des hyperparamètres...")
    # --- 8) Fit
    search.fit(X_train, y_train)

    print("Meilleurs paramètres :", search.best_params_)

    print("Évaluation du modèle sur le set de validation...")
    # --- 9) Évaluation finale
    y_pred = search.predict(X_val)
    score = f1_score(y_val, y_pred)

    print(classification_report(y_val, y_pred))
    print(f"F1 final = {score:.4f}")

    # --- 10) Sauvegarde propre
    features = list(X.columns)
    model_path = save_model(
        model=search.best_estimator_,
        model_name="xgboost_tfidf_svd",
        score=score,
        features=features,
        path="models"
    )

    print("Modèle sauvegardé dans :", model_path)


if __name__ == "__main__":
    main()

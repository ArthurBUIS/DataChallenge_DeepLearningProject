"""
Script d'entraînement pour XGBoost avec tuning hyperparamètres et CV stratifiée.
Usage:
    python scripts/train_xgboost_tuned.py
"""

import os
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import classification_report, f1_score
import numpy as np

from src.data.dataset import load_dataset
from src.data.preprocessing import add_account_age, add_hashtag_count
from src.models.xgboost import build_xgboost_model
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

    # --- 3) Traitement minimal des valeurs manquantes
    if "possibly_sensitive" in X.columns:
        X["possibly_sensitive"] = X["possibly_sensitive"].fillna(0).astype(int)
    if "is_quote_status" in X.columns:
        X["is_quote_status"] = X["is_quote_status"].fillna(0).astype(int)
    if "user.description" in X.columns:
        X["user.description"] = X["user.description"].fillna("")

    # --- 4) Split pour early stopping
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Sécurité supplémentaire pour texte et description
    for col in ["full_text", "user.description"]:
        X_train[col] = X_train[col].fillna("").astype(str)
        X_val[col] = X_val[col].fillna("").astype(str)

    print("Construction du modèle XGBoost...")
    # --- 5) Build modèle XGBoost
    base_model = build_xgboost_model()

    # --- 6) Définition de l'espace de recherche pour hyperparam tuning
    param_dist = {
        "clf__n_estimators": [200, 500, 800],
        "clf__max_depth": [3, 5, 7, 9],
        "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "clf__subsample": [0.6, 0.8, 1.0],
        "clf__colsample_bytree": [0.6, 0.8, 1.0],
    }

    # --- 7) Cross-validation stratifiée avec RandomizedSearch
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=20,  # nombre de combinaisons testées
        scoring="f1",
        cv=cv,
        verbose=2,
        n_jobs=1,
        random_state=42,
    )

    print("Démarrage du tuning des hyperparamètres...")
    # --- 8) Fit avec early stopping via eval_set
    search.fit(X_train, y_train)

    print("Meilleurs paramètres trouvés :", search.best_params_)

    print("Évaluation du modèle sur le set de validation...")
    # --- 9) Evaluation
    y_pred = search.predict(X_val)
    score = f1_score(y_val, y_pred)
    print(classification_report(y_val, y_pred))
    print(f"F1 = {score:.4f}")

    # --- 10) Save modèle et metadata
    features = list(X.columns)
    model_path = save_model(
        model=search.best_estimator_,
        model_name="xgboost_tuned",
        score=score,
        features=features,
        path="models"
    )
    print("Modèle sauvegardé dans :", model_path)


if __name__ == "__main__":
    main()

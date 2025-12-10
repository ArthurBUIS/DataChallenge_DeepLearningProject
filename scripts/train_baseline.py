"""
Script d'entraînement pour le baseline LogisticRegression.
Usage:
    python scripts/train_baseline.py
"""

import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

from src.data.dataset import load_dataset
from src.data.preprocessing import add_account_age, add_hashtag_count
from src.models.baseline import build_baseline_model
from src.utils.seed import set_seed
from src.utils.model_saving import save_model

def main():
    set_seed(42)

    # --- 1) Chargement
    X, y = load_dataset(split="train", version="essential")

    # --- 2) Feature engineering
    X = add_account_age(X)
    X = add_hashtag_count(X)

    # --- 3) Traitement minimal des valeurs manquantes (si pas déjà fait)
    if "possibly_sensitive" in X.columns:
        X["possibly_sensitive"] = X["possibly_sensitive"].fillna(0).astype(int)
    if "is_quote_status" in X.columns:
        X["is_quote_status"] = X["is_quote_status"].fillna(0).astype(int)
    if "user.description" in X.columns:
        X["user.description"] = X["user.description"].fillna("")

    # --- 4) Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # --- 4 bis) Ajout d'une sécurité en plus contre les valeurs manquantes
    X_train["full_text"] = X_train["full_text"].fillna("").astype(str)
    X_val["full_text"] = X_val["full_text"].fillna("").astype(str)

    X_train["user.description"] = X_train["user.description"].fillna("").astype(str)
    X_val["user.description"] = X_val["user.description"].fillna("").astype(str)

    # --- 5) Build & train
    model = build_baseline_model()
    model.fit(X_train, y_train)

    # --- 6) Eval
    y_pred = model.predict(X_val)
    score = f1_score(y_val, y_pred)
    print(classification_report(y_val, y_pred))
    print(f"F1 = {score:.4f}")

    # --- 7) Save model & metadata
    features = list(X.columns)
    model_path = save_model(
        model=model, model_name="baseline_logreg", score=score, features=features, path="models"
    )
    print("Modèle sauvegardé dans :", model_path)

if __name__ == "__main__":
    main()

"""
Entraînement manuel XGBoost (TF-IDF + SVD) avec boucle d'hyperparamètres.
Usage:
    python -m scripts.train_xgboost3_manual
"""

from xml.parsers.expat import model
import numpy as np
import copy
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

from src.data.dataset import load_dataset
from src.models.xgboost3 import build_xgboost_model
from src.utils.seed import set_seed
from src.utils.model_saving import save_model
import matplotlib.pyplot as plt
import time
import copy


def main():
    set_seed(42)

    print("Chargement du dataset FINAL...")
    X, y = load_dataset(split="train", version="final")

    # --- Sécurisation texte
    for col in ["full_text", "user.description"]:
        X[col] = X[col].fillna("").astype(str)

    # --- Split fixe pour comparabilité
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("Construction du modèle de base...")
    base_model = build_xgboost_model(
        n_components_text=300,
        n_components_desc=100,
    )

    # ============================================================
    # ✅ GRILLE MANUELLE D'HYPERPARAMÈTRES
    # ============================================================

    param_list = [
        {
            "n_estimators": 600,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 1,
        },
        {
            "n_estimators": 800,
            "max_depth": 7,
            "learning_rate": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 3,
        },
        {
            "n_estimators": 1000,
            "max_depth": 7,
            "learning_rate": 0.03,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 3,
        },
        {
            "n_estimators": 800,
            "max_depth": 9,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 5,
        },
    ]

    best_score = 0
    best_model = None
    best_params = None

    # ============================================================
    # ✅ BOUCLE D'ENTRAÎNEMENT MANUELLE ROBUSTE (SANS eval_set)
    # ============================================================

    for i, params in enumerate(param_list):
        print("\n" + "=" * 80)
        print(f"Modèle {i + 1}/{len(param_list)} avec paramètres :")
        print(params)

        model = copy.deepcopy(base_model)
        model.set_params(
            clf__n_estimators=params["n_estimators"],
            clf__max_depth=params["max_depth"],
            clf__learning_rate=params["learning_rate"],
            clf__subsample=params["subsample"],
            clf__colsample_bytree=params["colsample_bytree"],
            clf__min_child_weight=params["min_child_weight"],
            clf__eval_metric="logloss",
            clf__verbosity=1   # ✅ corrigé (remplace verbose)
        )

        print("Entraînement...")
        start_time = time.time()

        model.fit(X_train, y_train)

        train_time = time.time() - start_time
        print(f"⏱️ Temps d'entraînement : {train_time:.1f} s")
        print(f"🌲 Nombre d'arbres entraînés : {params['n_estimators']}")

        # ============================================================
        # ✅ ÉVALUATION
        # ============================================================

        print("Évaluation...")
        y_pred = model.predict(X_val)
        y_train_pred = model.predict(X_train)

        train_f1 = f1_score(y_train, y_train_pred)
        val_f1   = f1_score(y_val, y_pred)

        print(f"Train F1 = {train_f1:.4f}")
        print(f"Val   F1 = {val_f1:.4f}")

        score = val_f1

        print(classification_report(y_val, y_pred))
        print(f"F1 = {score:.4f}")

        if score > best_score:
            print("✅ Nouveau meilleur modèle trouvé !")
            best_score = score
            best_model = model
            best_params = params

    # ============================================================
    # ✅ SAUVEGARDE DU MEILLEUR MODÈLE
    # ============================================================

    print("\n" + "#" * 80)
    print("MEILLEUR MODÈLE FINAL")
    print("Score F1 :", best_score)
    print("Paramètres :", best_params)

    features = list(X.columns)
    model_path = save_model(
        model=best_model,
        model_name="xgboost_tfidf_svd_final_manual",
        score=best_score,
        features=features,
        path="models",
    )

    print("Modèle sauvegardé dans :", model_path)


if __name__ == "__main__":
    main()

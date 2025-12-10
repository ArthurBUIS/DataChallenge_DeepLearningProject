"""
Inference du modèle xgboost2 + génération de la submission Kaggle
"""

import pandas as pd
import joblib
import os
from datetime import datetime

from src.data.dataset import load_dataset


def main():
    # --- 1) Chargement du dataset Kaggle ESSENTIAL
    X_kaggle, _ = load_dataset(split="kaggle", version="final")
    X_kaggle_full, _ = load_dataset(split="kaggle", version="full")

    # Sécurité NaN texte
    X_kaggle["full_text"] = X_kaggle["full_text"].fillna("").astype(str)
    X_kaggle["user.description"] = X_kaggle["user.description"].fillna("").astype(str)

    # --- 2) Chargement du modèle
    model_path = "models/xgboost_tfidf_svd_final_manual_0.9263_20251209_031207.joblib"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")

    model = joblib.load(model_path)
    print("✅ Modèle chargé :", model_path)

    # --- 3) Prédictions
    y_pred = model.predict(X_kaggle)
    
    tweet_ids = X_kaggle_full['challenge_id'] 
    
    # --- 3bis) Probabilités (optionnel)
    y_proba = model.predict_proba(X_kaggle)[:, 1]
    proba_df = pd.DataFrame({
        "ID": tweet_ids,
        "Prediction_Probability": y_proba
    })
    os.makedirs("submissions", exist_ok=True)
    proba_path = f"submissions/prediction_probabilities_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    proba_df.to_csv(proba_path, index=False)
    print("✅ Probabilités sauvegardées :", proba_path)

    # --- 4) Création de la submission
    submission = pd.DataFrame({
        "ID": tweet_ids, 
        "Prediction": y_pred.astype(int)
    })

    # --- 5) Sauvegarde horodatée
    os.makedirs("submissions", exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"submissions/submission_{date_str}.csv"

    submission.to_csv(path, index=False)

    print("✅ Submission générée :", path)
    print(submission.head())


if __name__ == "__main__":
    main()
    
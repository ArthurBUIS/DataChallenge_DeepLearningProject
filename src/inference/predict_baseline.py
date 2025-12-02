"""
Inference du modèle baseline + génération de la submission Kaggle
"""

import pandas as pd
import joblib
import os
from datetime import datetime

from src.data.dataset import load_dataset


def main():
    # --- 1) Chargement du dataset Kaggle ESSENTIAL
    X_kaggle, _ = load_dataset(split="kaggle", version="essential")
    X_kaggle_full, _ = load_dataset(split="kaggle", version="full")

    # Sécurité NaN texte
    X_kaggle["full_text"] = X_kaggle["full_text"].fillna("").astype(str)
    X_kaggle["user.description"] = X_kaggle["user.description"].fillna("").astype(str)

    # --- 2) Chargement du modèle
    model_path = "models/baseline_logreg_0.8089_20251202_140939.joblib"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")

    model = joblib.load(model_path)
    print("✅ Modèle chargé :", model_path)

    # --- 3) Prédictions
    y_pred = model.predict(X_kaggle)
    
    tweet_ids = X_kaggle_full['challenge_id'] 

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
    
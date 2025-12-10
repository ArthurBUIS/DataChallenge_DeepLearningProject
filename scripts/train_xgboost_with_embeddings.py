# scripts/train_xgboost_with_embeddings.py
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from xgboost import XGBClassifier

from src.utils.seed import set_seed
from src.utils.model_saving import save_model
import xgboost as xgb


def load_features_and_embeddings():
    X_df = pd.read_csv("data/processed/X_train_final.csv")
    y = pd.read_csv("data/processed/y_train_final.csv").iloc[:, 0].values.ravel()

    emb = np.load("data/processed/embeddings/X_train_text_emb.npy")

    with open("data/processed/embeddings/text_embedding_features.json") as f:
        emb_features = json.load(f)

    text_cols = ["full_text", "user.description"]
    numeric_cols = [c for c in X_df.columns if c not in text_cols]

    X_num = X_df[numeric_cols].fillna(0).astype(float).values

    X_all = np.hstack([X_num, emb.astype(np.float32)])

    full_feature_names = numeric_cols + emb_features
    return X_all, y, full_feature_names


def main():
    set_seed(42)

    print("Loading data + embeddings...")
    X_all, y, feature_names = load_features_and_embeddings()

    print("Final feature matrix:", X_all.shape)

    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y, test_size=0.2, stratify=y, random_state=42
    )

    model = XGBClassifier(
        n_estimators=1200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=1,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="gpu_hist",
        predictor="gpu_predictor",
        random_state=42,
        n_jobs=-1,
    )

    eval_set = [(X_train, y_train), (X_val, y_val)]

    print("Training model...")
    t0 = time.time()

    model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        verbose=50,
        callbacks=[
            xgb.callback.EarlyStopping(
                rounds=50,
                save_best=True,
            )
        ],
    )
    print(f"Training time: {time.time() - t0:.1f}s")

    result = model.evals_result()

    train_loss = result["validation_0"]["logloss"]
    val_loss = result["validation_1"]["logloss"]

    plt.figure(figsize=(8, 5))
    plt.plot(train_loss, label="train")
    plt.plot(val_loss, label="val")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("models/training_curve_xgb_sbert.png")
    plt.close()

    y_pred = model.predict(X_val)
    score = f1_score(y_val, y_pred)

    print(classification_report(y_val, y_pred))
    print("F1 =", score)

    model_path = save_model(
        model=model,
        model_name="xgboost_sbert",
        score=score,
        features=feature_names,
        path="models",
    )

    # Sauvegarde propre des features complètes
    with open("models/features_xgboost_sbert.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    print("Model saved to:", model_path)
    print("Features saved to: models/features_xgboost_sbert.json")


if __name__ == "__main__":
    main()

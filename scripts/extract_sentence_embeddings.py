# scripts/extract_sentence_embeddings.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import json


def extract_and_save(
    model_name="paraphrase-multilingual-MiniLM-L12-v2",
    data_dir="data/processed",
    train_csv="X_train_final.csv",
    kaggle_csv="X_kaggle_final.csv",
    out_dir="data/processed/embeddings",
    batch_size=128,
    device="cuda",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading SentenceTransformer:", model_name)
    model = SentenceTransformer(model_name, device=device)

    def encode_column(texts, name):
        embeddings = []
        print(f"Encoding {name} ({len(texts)} samples)")
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i : i + batch_size]
            emb = model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            embeddings.append(emb)
        return np.vstack(embeddings)

    def encode_pair(df):
        texts1 = df["full_text"].fillna("").astype(str).tolist()
        texts2 = df["user.description"].fillna("").astype(str).tolist()

        emb1 = encode_column(texts1, "full_text")
        emb2 = encode_column(texts2, "user.description")

        emb = np.hstack([emb1, emb2])

        # Sauvegarde des noms de dimensions
        emb_dim = emb.shape[1]
        feature_names = (
            [f"emb_full_text_{i}" for i in range(emb1.shape[1])]
            + [f"emb_user_desc_{i}" for i in range(emb2.shape[1])]
        )

        return emb, feature_names

    # ============ TRAIN ============
    train_path = Path(data_dir) / train_csv
    if train_path.exists():
        df_train = pd.read_csv(train_path)
        emb_train, feature_names = encode_pair(df_train)

        np.save(out_dir / "X_train_text_emb.npy", emb_train)

        with open(out_dir / "text_embedding_features.json", "w") as f:
            json.dump(feature_names, f, indent=2)

        print("Saved:", out_dir / "X_train_text_emb.npy")
    else:
        print("Train CSV not found")

    # ============ KAGGLE ============
    kaggle_path = Path(data_dir) / kaggle_csv
    if kaggle_path.exists():
        df_kaggle = pd.read_csv(kaggle_path)
        emb_kaggle, _ = encode_pair(df_kaggle)
        np.save(out_dir / "X_kaggle_text_emb.npy", emb_kaggle)
        print("Saved:", out_dir / "X_kaggle_text_emb.npy")
    else:
        print("Kaggle CSV not found")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--out_dir", default="data/processed/embeddings")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    extract_and_save(
        model_name=args.model,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        device=args.device,
    )

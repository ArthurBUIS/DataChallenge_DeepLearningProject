import numpy as np
import pandas as pd
from pathlib import Path

# ========================
# PARAMÈTRES
# ========================

CSV_PATH = "data/processed/X_train_final.csv"
CSV_PATH_KAGGLE = "data/processed/X_kaggle_final.csv"
NPY_PATH = "data/processed/embeddings/X_train_text_emb.npy"
NPY_PATH_KAGGLE = "data/processed/embeddings/X_kaggle_text_emb.npy"
OUT_PATH = "data/processed/X_train_embeddings.csv"
OUT_PATH_KAGGLE = "data/processed/X_kaggle_embeddings.csv"

# ========================
# CHARGEMENT
# ========================

print("Chargement CSV...")
df = pd.read_csv(CSV_PATH)
df_kaggle = pd.read_csv(CSV_PATH_KAGGLE)



print("Chargement embeddings NPY...")
embeddings = np.load(NPY_PATH)
embeddings_kaggle = np.load(NPY_PATH_KAGGLE)

assert len(df) == embeddings.shape[0], \
    f"Incohérence tailles : df={len(df)} vs emb={embeddings.shape[0]}"
assert len(df_kaggle) == embeddings_kaggle.shape[0], \
    f"Incohérence tailles : df_kaggle={len(df_kaggle)} vs emb_kaggle={embeddings_kaggle.shape[0]}"

# ========================
# CONVERSION EMBEDDINGS → COLONNES
# ========================

print("Conversion embeddings → DataFrame...")
emb_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
emb_cols_kaggle = [f"emb_{i}" for i in range(embeddings_kaggle.shape[1])]
df_emb = pd.DataFrame(embeddings, columns=emb_cols)
df_emb_kaggle = pd.DataFrame(embeddings_kaggle, columns=emb_cols_kaggle)

# ========================
# CONCATÉNATION
# ========================

df_final = pd.concat([df.reset_index(drop=True), df_emb], axis=1)
df_final_kaggle = pd.concat([df_kaggle.reset_index(drop=True), df_emb_kaggle], axis=1)

print("Shape final :", df_final.shape)
print("Shape final kaggle :", df_final_kaggle.shape)

# ========================
# SAUVEGARDE
# ========================

Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
df_final.to_csv(OUT_PATH, index=False)
Path(OUT_PATH_KAGGLE).parent.mkdir(parents=True, exist_ok=True)
df_final_kaggle.to_csv(OUT_PATH_KAGGLE, index=False)

print("✅ Dataset embeddings construit :", OUT_PATH)
print("✅ Dataset embeddings kaggle construit :", OUT_PATH_KAGGLE)

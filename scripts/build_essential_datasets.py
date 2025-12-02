import pandas as pd

from src.data.preprocessing import add_account_age, add_hashtag_count


ESSENTIAL_FEATURES = [
    "full_text",
    "extended_tweet.entities.hashtags",
    "possibly_sensitive",
    "is_quote_status",
    "user.statuses_count",
    "user.favourites_count",
    "user.description",
    "user.created_at",
]


def resolve_full_text(X):
    if "full_text" not in X.columns:
        if "extended_tweet.full_text" in X.columns:
            X["full_text"] = X["extended_tweet.full_text"]
        elif "text" in X.columns:
            X["full_text"] = X["text"]
        else:
            raise ValueError("Impossible de reconstruire full_text.")
    return X


def main():
    print("Loading raw datasets...")

    X_train = pd.read_csv("data/raw/X_train.csv")
    y_train = pd.read_csv("data/raw/y_train.csv").iloc[:, 0]
    X_kaggle = pd.read_csv("data/raw/X_kaggle.csv")

    print("Reconstructing the full text")
    # --- Reconstruction full_text
    X_train = resolve_full_text(X_train)
    X_kaggle = resolve_full_text(X_kaggle)

    print("Selecting essential features")
    # --- Sélection des features essentielles
    X_train = X_train[ESSENTIAL_FEATURES].copy()
    X_kaggle = X_kaggle[ESSENTIAL_FEATURES].copy()

    # --- Feature engineering
    X_train = add_account_age(X_train)
    X_kaggle = add_account_age(X_kaggle)

    X_train = add_hashtag_count(X_train)
    X_kaggle = add_hashtag_count(X_kaggle)

    print("Cleaning types / NaN values")
    # --- Nettoyage types / NaN
    X_train["possibly_sensitive"] = X_train["possibly_sensitive"].fillna(0).astype(int)
    X_kaggle["possibly_sensitive"] = X_kaggle["possibly_sensitive"].fillna(0).astype(int)

    X_train["is_quote_status"] = X_train["is_quote_status"].fillna(0).astype(int)
    X_kaggle["is_quote_status"] = X_kaggle["is_quote_status"].fillna(0).astype(int)

    X_train["user.description"] = X_train["user.description"].fillna("")
    X_kaggle["user.description"] = X_kaggle["user.description"].fillna("")

    # --- Sauvegarde
    print("Saving essential datasets...")

    X_train.to_csv("data/processed/X_train_essential.csv", index=False)
    y_train.to_csv("data/processed/y_train_essential.csv", index=False)
    X_kaggle.to_csv("data/processed/X_kaggle_essential.csv", index=False)

    print("✅ Essential datasets generated in data/processed/")


if __name__ == "__main__":
    main()

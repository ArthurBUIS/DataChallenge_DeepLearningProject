"""
XGBoost model definition: TF-IDF on tweet text + TF-IDF on user.description
+ numeric features scaled, then XGBoost classifier.
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier


def build_xgboost_model(
    max_text_features=30000,
    max_desc_features=10000,
    ngram_range_text=(1, 2),
    ngram_range_desc=(1, 2),
    xgb_params=None,
):
    """
    Retourne un pipeline sklearn prêt à fit() :
      - préprocesseur (TF-IDF pour texte + TF-IDF pour description + scaler numérique)
      - classifieur XGBClassifier
    """

    # colonnes attendues dans le DataFrame d'entrée
    text_col = "full_text"
    desc_col = "user.description"
    num_features = [
        "possibly_sensitive",
        "is_quote_status",
        "user.statuses_count",
        "user.favourites_count",
        "account_age_days",
        "n_hashtags",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "tweet_text",
                TfidfVectorizer(
                    max_features=max_text_features,
                    ngram_range=ngram_range_text,
                    strip_accents="unicode",
                    lowercase=True,
                ),
                text_col,
            ),
            (
                "user_desc",
                TfidfVectorizer(
                    max_features=max_desc_features,
                    ngram_range=ngram_range_desc,
                    strip_accents="unicode",
                    lowercase=True,
                ),
                desc_col,
            ),
            ("numeric", StandardScaler(), num_features),
        ],
        remainder="drop",
    )

    # paramètres par défaut si aucun n'est fourni
    if xgb_params is None:
        xgb_params = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist",  # Utilisation du GPU si disponible
            "device": "cuda",
        }

    clf = XGBClassifier(**xgb_params)

    pipeline = Pipeline([("preprocess", preprocessor), ("clf", clf)])
    print("XGBoost model built successfully")
    return pipeline


"""
XGBoost model definition: TF-IDF on tweet text and on user.description
+ SVD, then XGBoost classifier.
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from xgboost import XGBClassifier


def build_xgboost_model(
    max_text_features=40000,
    max_desc_features=15000,
    ngram_range_text=(1, 2),
    ngram_range_desc=(1, 2),
    n_components_text=300,
    n_components_desc=100,
    xgb_params=None,
):
    """
    XGBoost avec :
      - TF-IDF + SVD sur full_text
      - TF-IDF + SVD sur user.description
      - Features numériques scalées
    """

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

    text_pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_text_features,
                    ngram_range=ngram_range_text,
                    strip_accents="unicode",
                    lowercase=True,
                    min_df=3,
                ),
            ),
            ("svd", TruncatedSVD(n_components=n_components_text, random_state=42)),
        ]
    )

    desc_pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_desc_features,
                    ngram_range=ngram_range_desc,
                    strip_accents="unicode",
                    lowercase=True,
                    min_df=2,
                ),
            ),
            ("svd", TruncatedSVD(n_components=n_components_desc, random_state=42)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("tweet_text", text_pipeline, text_col),
            ("user_desc", desc_pipeline, desc_col),
            ("numeric", StandardScaler(), num_features),
        ],
        remainder="drop",
    )

    if xgb_params is None:
        xgb_params = {
            "n_estimators": 800,
            "max_depth": 7,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist",
            "device": "cuda",
        }

    clf = XGBClassifier(**xgb_params)

    pipeline = Pipeline(
        [
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )

    print("✅ XGBoost TF-IDF + SVD model built successfully")
    return pipeline

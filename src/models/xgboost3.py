from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from xgboost import XGBClassifier


def build_xgboost_model(
    n_components_text=300,
    n_components_desc=100,
    max_features_text=30000,
    max_features_desc=10000,
    xgb_params=None,
):
    """
    Pipeline :
    - TF-IDF + SVD sur full_text
    - TF-IDF + SVD sur user.description
    - Features numériques passées telles quelles
    - XGBoost en sortie
    """

    text_col = "full_text"
    desc_col = "user.description"

    # ✅ Toutes les colonnes sauf les deux textes = numériques
    numeric_features = [
        "user.favourites_count",
        "user.listed_count",
        "user.statuses_count",
        "user.profile_use_background_image",
        "user.default_profile",
        "user.geo_enabled",
        "user.profile_background_tile",
        "all_user_mentions_count",
        "quoted_status.favorite_count",
        "quoted_status.user.favourites_count",
        "others_1",
        "others_0",
        "others",
        "Twitter Web App",
        "Twitter for iPhone",
        "Hootsuite Inc.",
        "TweetDeck",
        "Twitter for Android",
        "Echobox",
        "has_photo",
        "has_video",
        "hashtag_covid19",
        "favourites_per_status",
        "listed_per_status",
    ]

    # --- Pipeline TF-IDF + SVD pour le texte du tweet
    tweet_text_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features_text,
            ngram_range=(1, 2),
            strip_accents="unicode",
            lowercase=True,
        )),
        ("svd", TruncatedSVD(
            n_components=n_components_text,
            random_state=42
        )),
    ])

    # --- Pipeline TF-IDF + SVD pour la description utilisateur
    user_desc_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features_desc,
            ngram_range=(1, 2),
            strip_accents="unicode",
            lowercase=True,
        )),
        ("svd", TruncatedSVD(
            n_components=n_components_desc,
            random_state=42
        )),
    ])

    # --- Assemblage final
    preprocessor = ColumnTransformer(
        transformers=[
            ("tweet_text", tweet_text_pipe, text_col),
            ("user_desc", user_desc_pipe, desc_col),
            ("num", "passthrough", numeric_features),
        ],
        remainder="drop",
        n_jobs=1
    )

    # --- Paramètres XGBoost par défaut (GPU safe)
    if xgb_params is None:
        xgb_params = {
            "n_estimators": 600,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 1,
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist",
            "device": "cuda",   # ✅ ok avec TF-IDF + numpy sparse
        }

    clf = XGBClassifier(**xgb_params)

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("clf", clf),
    ])

    print("✅ XGBoost + TF-IDF + SVD model built successfully")
    return pipeline

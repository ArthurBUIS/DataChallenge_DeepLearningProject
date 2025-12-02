"""
Baseline model definition: TF-IDF on tweet text + TF-IDF on user.description
+ numeric features scaled, then LogisticRegression.
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def build_baseline_model(
    max_text_features=30000,
    max_desc_features=10000,
    ngram_range_text=(1, 2),
    ngram_range_desc=(1, 2),
):
    """
    Retourne un pipeline sklearn prêt à fit() :
      - préprocesseur (TF-IDF pour texte + TF-IDF pour description + scaler numérique)
      - classifieur LogisticRegression
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

    clf = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    pipeline = Pipeline([("preprocess", preprocessor), ("clf", clf)])
    return pipeline

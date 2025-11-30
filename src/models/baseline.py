from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def build_baseline_model():

    text_features = ["full_text", "user.description"]
    num_features = [
        "possibly_sensitive",
        "is_quote_status",
        "user.statuses_count",
        "user.favourites_count",
        "account_age_days",
        "n_hashtags"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(max_features=30000, ngram_range=(1, 2)),
             "full_text"),
            ("desc", TfidfVectorizer(max_features=10000, ngram_range=(1, 2)),
             "user.description"),
            ("num", StandardScaler(), num_features)
        ]
    )

    clf = LogisticRegression(max_iter=5000, class_weight="balanced")

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", clf)
    ])

    return pipeline

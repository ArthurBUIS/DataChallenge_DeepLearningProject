"""
Fonctions pures de feature engineering :
- add_account_age: dérive account_age_days à partir de user.created_at
- add_hashtag_count: dérive n_hashtags à partir du champ extended_tweet.entities.hashtags
"""

from datetime import datetime, timezone
import pandas as pd

def add_account_age(df, date_col="user.created_at"):
    df = df.copy()

    # Conversion en datetime
    dates = pd.to_datetime(df[date_col], errors="coerce", utc=True)

    # Maintenant tout est tz-aware en UTC
    now = datetime.now(timezone.utc)

    df["account_age_days"] = (now - dates).dt.days
    df["account_age_days"] = df["account_age_days"].fillna(0).astype(int)

    return df


def add_hashtag_count(df, hashtag_col="extended_tweet.entities.hashtags"):
    df = df.copy()
    def safe_count(x):
        # on gère les cas où la colonne est NaN, '', déjà list, ou str like "[]"
        try:
            if pd.isna(x):
                return 0
            if isinstance(x, list):
                return len(x)
            if isinstance(x, str):
                # si c'est un JSON-like list string, on l'évalue prudemment
                # NOTE: on suppose que le format est similaire à ton notebook ("[...]")
                if x.strip().startswith("["):
                    # utilisation de eval ici pour compatibilité avec ton ancien notebook
                    # Si tu veux être plus sûr, remplacer par json.loads après nettoyage.
                    return len(eval(x))
                else:
                    return 0
            return 0
        except Exception:
            return 0

    df["n_hashtags"] = df[hashtag_col].apply(safe_count)
    return df

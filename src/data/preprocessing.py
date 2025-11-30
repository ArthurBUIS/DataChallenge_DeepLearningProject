import pandas as pd
from datetime import datetime

def add_account_age(df):
    now = datetime.utcnow()
    df["account_age_days"] = pd.to_datetime(
        df["user.created_at"], errors="coerce"
    ).apply(lambda d: (now - d).days if pd.notnull(d) else 0)
    return df

def add_hashtag_count(df):
    df["n_hashtags"] = df["extended_tweet.entities.hashtags"].apply(
        lambda x: len(eval(x)) if isinstance(x, str) and x.startswith("[") else 0
    )
    return df

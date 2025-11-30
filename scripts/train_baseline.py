from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

from src.data.dataset import load_dataset
from src.data.preprocessing import add_account_age, add_hashtag_count
from src.models.baseline import build_baseline_model
from src.utils.seed import set_seed
from src.utils.model_saving import save_model

set_seed(42)

# Chargement des données
X, y = load_dataset("train", feature_set="baseline")

# Feature engineering
X = add_account_age(X)
X = add_hashtag_count(X)

# Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Modèle
model = build_baseline_model()
model.fit(X_train, y_train)

# Évaluation
y_pred = model.predict(X_val)
score = f1_score(y_val, y_pred)

print(classification_report(y_val, y_pred))
print("F1 score =", score)

# Sauvegarde
model_path = save_model(
    model,
    model_name="baseline_logreg",
    score=score,
    features=list(X.columns)
)

print("Modèle sauvegardé dans :", model_path)

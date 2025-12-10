import joblib
import json
import os
from datetime import datetime

def save_model(model, model_name, score, features, path="models"):
    os.makedirs(path, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{score:.4f}_{ts}.joblib"
    model_path = os.path.join(path, filename)

    joblib.dump(model, model_path)

    metadata = {
        "model_name": model_name,
        "score": score,
        "features": features,
        "date": datetime.now().isoformat()
    }

    with open(model_path.replace(".joblib", ".json"), "w") as f:
        json.dump(metadata, f, indent=4)

    return model_path

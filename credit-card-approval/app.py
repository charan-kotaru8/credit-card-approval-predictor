from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
METADATA_PATH = MODELS_DIR / "metadata.json"

app = Flask(__name__)


def load_artifacts():
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    best_model = joblib.load(BEST_MODEL_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return preprocessor, best_model, metadata


# Cache on startup
preprocessor, best_model, metadata = load_artifacts()


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/predict")
def predict_form():
    return render_template(
        "predict.html",
        metadata=metadata,
    )


@app.post("/predict")
def predict():
    # Build one-row dataframe matching model input schema expected by preprocessor
    numeric = {
        "CNT_CHILDREN": float(request.form.get("CNT_CHILDREN", 0) or 0),
        "AMT_INCOME_TOTAL": float(request.form.get("AMT_INCOME_TOTAL", 0) or 0),
        "AGE_YEARS": float(request.form.get("AGE_YEARS", 0) or 0),
        "EMPLOYED_YEARS": float(request.form.get("EMPLOYED_YEARS", 0) or 0),
        "CNT_FAM_MEMBERS": float(request.form.get("CNT_FAM_MEMBERS", 0) or 0),
    }

    binary = {
        "FLAG_WORK_PHONE": int(request.form.get("FLAG_WORK_PHONE", 0) or 0),
        "FLAG_PHONE": int(request.form.get("FLAG_PHONE", 0) or 0),
        "FLAG_EMAIL": int(request.form.get("FLAG_EMAIL", 0) or 0),
        # original model expects these engineered bins
        "CODE_GENDER_BIN": int(request.form.get("CODE_GENDER_BIN", 0) or 0),
        "FLAG_OWN_CAR_BIN": int(request.form.get("FLAG_OWN_CAR_BIN", 0) or 0),
        "FLAG_OWN_REALTY_BIN": int(request.form.get("FLAG_OWN_REALTY_BIN", 0) or 0),
    }

    cat = {}
    for c in metadata.get("categorical_features", []):
        cat[c] = request.form.get(c)

    row = {**numeric, **binary, **cat}

    import pandas as pd

    X = pd.DataFrame([row])
    X_t = preprocessor.transform(X)
    prob_rejected = 1.0 - best_model.predict_proba(X_t)[:, 1][0]
    prob_approved = 1.0 - prob_rejected

    pred = int(prob_approved >= 0.5)
    label = "Approved" if pred == 1 else "Rejected"

    # Store simple in-memory history in session-less file
    history_path = BASE_DIR / "models" / "prediction_history.json"
    history_path.parent.mkdir(exist_ok=True, parents=True)
    try:
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if not isinstance(history, list):
            history = []
    except Exception:
        history = []

    history.insert(
        0,
        {
            "label": label,
            "prob_approved": float(prob_approved),
            "prob_rejected": float(prob_rejected),
            "input": row,
        },
    )
    history = history[:50]
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    return render_template(
        "result.html",
        label=label,
        prob_approved=prob_approved,
        prob_rejected=prob_rejected,
    )


@app.get("/history")
def history():
    history_path = BASE_DIR / "models" / "prediction_history.json"
    try:
        history = json.loads(history_path.read_text(encoding="utf-8"))
    except Exception:
        history = []
    return render_template("history.html", history=history)


if __name__ == "__main__":
    app.run(debug=True)


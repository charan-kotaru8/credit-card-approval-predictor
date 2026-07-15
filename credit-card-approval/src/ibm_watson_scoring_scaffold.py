"""IBM Watson Machine Learning deployment scaffolding (documentation helper).

This file does not execute deployment by itself; it provides a template
for how to package your artifacts and call a WML scoring endpoint.

Artifacts expected in models/:
- preprocessor.pkl
- best_model.pkl
- metadata.json

A common approach is to build a single scoring wrapper object.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"


def load_local_assets():
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")
    best_model = joblib.load(MODELS_DIR / "best_model.pkl")
    metadata = json.loads((MODELS_DIR / "metadata.json").read_text(encoding="utf-8"))
    return preprocessor, best_model, metadata


def predict_payload(payload: dict) -> dict:
    """Score a single applicant.

    payload example keys must match the web form rows created in app.py.
    """
    preprocessor, best_model, metadata = load_local_assets()

    numeric = {
        "CNT_CHILDREN": float(payload.get("CNT_CHILDREN", 0) or 0),
        "AMT_INCOME_TOTAL": float(payload.get("AMT_INCOME_TOTAL", 0) or 0),
        "AGE_YEARS": float(payload.get("AGE_YEARS", 0) or 0),
        "EMPLOYED_YEARS": float(payload.get("EMPLOYED_YEARS", 0) or 0),
        "CNT_FAM_MEMBERS": float(payload.get("CNT_FAM_MEMBERS", 0) or 0),
    }

    binary = {
        "FLAG_WORK_PHONE": int(payload.get("FLAG_WORK_PHONE", 0) or 0),
        "FLAG_PHONE": int(payload.get("FLAG_PHONE", 0) or 0),
        "FLAG_EMAIL": int(payload.get("FLAG_EMAIL", 0) or 0),
        "CODE_GENDER_BIN": int(payload.get("CODE_GENDER_BIN", 0) or 0),
        "FLAG_OWN_CAR_BIN": int(payload.get("FLAG_OWN_CAR_BIN", 0) or 0),
        "FLAG_OWN_REALTY_BIN": int(payload.get("FLAG_OWN_REALTY_BIN", 0) or 0),
    }

    cat = {c: payload.get(c) for c in metadata.get("categorical_features", [])}

    row = {**numeric, **binary, **cat}
    X = pd.DataFrame([row])
    X_t = preprocessor.transform(X)

    prob_rejected = 1.0 - best_model.predict_proba(X_t)[:, 1][0]
    prob_approved = 1.0 - prob_rejected

    pred = int(prob_approved >= 0.5)
    label = "Approved" if pred == 1 else "Rejected"

    return {
        "label": label,
        "prob_approved": float(prob_approved),
        "prob_rejected": float(prob_rejected),
    }


if __name__ == "__main__":
    # Simple local smoke test
    sample = {
        "CNT_CHILDREN": 0,
        "AMT_INCOME_TOTAL": 50000,
        "AGE_YEARS": 30,
        "EMPLOYED_YEARS": 2,
        "CNT_FAM_MEMBERS": 2,
        "FLAG_WORK_PHONE": 1,
        "FLAG_PHONE": 0,
        "FLAG_EMAIL": 1,
        "CODE_GENDER_BIN": 0,
        "FLAG_OWN_CAR_BIN": 1,
        "FLAG_OWN_REALTY_BIN": 1,
    }
    print(predict_payload(sample))


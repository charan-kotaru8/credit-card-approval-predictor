"""Train preprocessing + models for credit card approval prediction.

Artifacts written to models/:
- preprocessor.pkl
- metadata.json
- best_model.pkl
- model_comparison.csv
- metrics tables + plots in notebooks/figures/

Run:
  python -m src.train_models
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


DATA_URL = (
    "https://raw.githubusercontent.com/bmariasirine/Credit_Card_Approval_Prediction/main/datasets.zip"
)


@dataclass
class Config:
    base_dir: Path
    data_dir: Path
    models_dir: Path
    figs_dir: Path

    def __init__(self):
        self.base_dir = Path(__file__).resolve().parents[1]
        self.data_dir = self.base_dir / "data"
        self.models_dir = self.base_dir / "models"
        self.figs_dir = self.base_dir / "notebooks" / "figures"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.figs_dir.mkdir(parents=True, exist_ok=True)


def _download_and_extract(url: str, out_dir: Path) -> Path:
    """Download datasets.zip and extract into out_dir.

    Returns path to extracted folder (out_dir)."""
    import zipfile
    import urllib.request

    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "datasets.zip"
    if not zip_path.exists():
        print(f"Downloading dataset from {url} -> {zip_path}")
        urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Extract only if CSVs not already present
        app_csv = out_dir / "application_record.csv"
        cred_csv = out_dir / "credit_record.csv"
        if not app_csv.exists() or not cred_csv.exists():
            print("Extracting datasets.zip...")
            zf.extractall(out_dir)

    return out_dir


def build_dataset(cfg: Config) -> pd.DataFrame:
    extracted = _download_and_extract(DATA_URL, cfg.data_dir)
    # dataset zip extracts into a nested folder in this mirror
    app_path = extracted / "application_record.csv"
    cred_path = extracted / "credit_record.csv"
    if not app_path.exists() or not cred_path.exists():
        nested_app = extracted / "datasets" / "application_record.csv"
        nested_cred = extracted / "datasets" / "credit_record.csv"
        if nested_app.exists() and nested_cred.exists():
            app_path = nested_app
            cred_path = nested_cred

    app = pd.read_csv(app_path)
    cred = pd.read_csv(cred_path)

    # Binary target: high-risk if ever status in {2,3,4,5}
    high_risk_status = {2, 3, 4, 5}
    # STATUS codes come in as strings (e.g., '0','2','5','X','C').
    status_numeric = pd.to_numeric(cred["STATUS"], errors="coerce")
    cred_flag = cred.assign(
        high_risk=status_numeric.isin(high_risk_status).astype(int)
    )


    # target=0 rejected if any high-risk, else approved => target=1
    # Ensure we keep IDs that have at least one application record
    target_df = cred_flag.groupby("ID", as_index=False)["high_risk"].max()
    target_df["TARGET"] = (1 - target_df["high_risk"]).astype(int)
    target_df = target_df[["ID", "TARGET"]]

    # Quick sanity: avoid training on a degenerate dataset
    # (happens if merge fails or STATUS types are unexpected)
    # If degenerate, raise early with counts.
    tgt_counts = target_df["TARGET"].value_counts(dropna=False).to_dict()
    if len(tgt_counts) < 2:
        raise RuntimeError(f"Degenerate TARGET distribution after engineering: {tgt_counts}")


    # Merge onto application_record: inner join
    df = app.merge(target_df, on="ID", how="inner")

    # Remove duplicates excluding ID
    feature_cols = [c for c in df.columns if c != "ID" and c != "TARGET"]
    df = df.drop_duplicates(subset=feature_cols, keep="first")
    return df


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()

    # Missing OCCUPATION_TYPE (~30%) -> Unknown
    if "OCCUPATION_TYPE" in df.columns:
        df["OCCUPATION_TYPE"] = df["OCCUPATION_TYPE"].fillna("Unknown")

    # Convert days to years; handle sentinel 365243 => 0
    sentinel = 365243
    df["AGE_YEARS"] = (df["DAYS_BIRTH"].clip(lower=0) / 365.25).astype(float)
    df["EMPLOYED_YEARS"] = df["DAYS_EMPLOYED"].apply(
        lambda x: 0.0 if x == sentinel else float(abs(x) / 365.25)
    )

    # Binary encodings (0/1)
    gender_map = {"M": 0, "F": 1}
    df["CODE_GENDER_BIN"] = df["CODE_GENDER"].map(gender_map).fillna(0).astype(int)

    car_map = {"N": 0, "Y": 1}
    realty_map = {"N": 0, "Y": 1}
    df["FLAG_OWN_CAR_BIN"] = df["FLAG_OWN_CAR"].map(car_map).fillna(0).astype(int)
    df["FLAG_OWN_REALTY_BIN"] = df["FLAG_OWN_REALTY"].map(realty_map).fillna(0).astype(int)

    # Drop ID and constant FLAG_MOBIL (no signal) and raw days columns
    drop_cols = [
        "ID",
        "TARGET",  # keep target separately
        "FLAG_MOBIL",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "CODE_GENDER",
        "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY",
    ]

    y = df["TARGET"].astype(int).values
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    metadata = {
        "numeric_features": [
            "CNT_CHILDREN",
            "AMT_INCOME_TOTAL",
            "AGE_YEARS",
            "EMPLOYED_YEARS",
            "CNT_FAM_MEMBERS",
        ],
        "binary_features": [
            "FLAG_WORK_PHONE",
            "FLAG_PHONE",
            "FLAG_EMAIL",
            "FLAG_MOBIL" if "FLAG_MOBIL" in X.columns else "FLAG_WORK_PHONE",
            "FLAG_OWN_CAR_BIN",
            "FLAG_OWN_REALTY_BIN",
            "CODE_GENDER_BIN",
        ],
        "categorical_features": [
            "NAME_INCOME_TYPE",
            "NAME_EDUCATION_TYPE",
            "NAME_FAMILY_STATUS",
            "NAME_HOUSING_TYPE",
            "OCCUPATION_TYPE",
        ],
        "categorical_options": {},
    }

    # Ensure binary feature columns actually exist
    binary_cols = [
        c
        for c in [
            "CODE_GENDER_BIN",
            "FLAG_OWN_CAR_BIN",
            "FLAG_OWN_REALTY_BIN",
            "FLAG_WORK_PHONE",
            "FLAG_PHONE",
            "FLAG_EMAIL",
            "FLAG_MOBIL",
        ]
        if c in X.columns
    ]
    metadata["binary_features"] = binary_cols

    # categorical options for later UI
    for c in metadata["categorical_features"]:
        if c in X.columns:
            metadata["categorical_options"][c] = sorted(
                [str(v) for v in X[c].dropna().unique().tolist()]
            )

    return X, y, metadata


def build_preprocessor(metadata: dict) -> ColumnTransformer:
    numeric_features = metadata["numeric_features"]
    binary_features = metadata["binary_features"]
    categorical_features = metadata["categorical_features"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    # Binary features are already 0/1 but still impute just in case
    binary_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("bin", binary_transformer, binary_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor


def _threshold_metrics(y_true, y_prob, threshold=0.5):
    from sklearn.metrics import f1_score, precision_recall_fscore_support, recall_score

    y_pred = (y_prob >= threshold).astype(int)
    # minority class is 0 (Rejected)
    p = precision_recall_fscore_support(y_true, y_pred, labels=[0], zero_division=0)
    precision_min, recall_min, f1_min, _ = p

    return {
        "precision_minority": float(precision_min[0]),
        "recall_minority": float(recall_min[0]),
        "f1_minority": float(f1_min[0]),
        "f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def plot_confusion_matrix(est_name: str, y_true, y_pred, fig_path: Path):
    plt.figure(figsize=(5, 4))
    cm_disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["Rejected(0)", "Approved(1)"],
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )
    cm_disp.ax_.set_title(f"Confusion Matrix - {est_name}")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()


def main():
    cfg = Config()
    df = build_dataset(cfg)
    X, y, metadata = engineer_features(df)

    # Create processed split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor = build_preprocessor(metadata)

    # Save metadata with categorical options
    metadata_path = cfg.models_dir / "metadata.json"
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    preprocessor_path = cfg.models_dir / "preprocessor.pkl"

    # Fit preprocessor only on train
    preprocessor.fit(X_train)
    joblib.dump(preprocessor, preprocessor_path)

    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    # Models: class-balanced weighting for minority class (0)
    # We'll use scale_pos_weight for xgb and class_weight for sklearn models.
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale_pos_weight = neg / max(pos, 1)

    models = {
        "logreg": LogisticRegression(
            max_iter=2000, class_weight="balanced", solver="lbfgs"
        ),
        "decision_tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=42,
        ),
    }

    results = []
    best = None
    best_f1_min = -1

    # Cross-validation (stratified)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        # Fit
        model.fit(X_train_t, y_train)

        # Test metrics
        prob = model.predict_proba(X_test_t)[:, 1]
        pred = (prob >= 0.5).astype(int)

        roc = roc_auc_score(y_test, prob)
        report = classification_report(y_test, pred, output_dict=True, zero_division=0)
        macro_f1 = report["macro avg"]["f1-score"]

        # Minority class metrics for label 0
        f1_min = report["0"]["f1-score"]
        precision_min = report["0"]["precision"]
        recall_min = report["0"]["recall"]

        plot_confusion_matrix(
            name,
            y_test,
            pred,
            cfg.figs_dir / f"confusion_matrix_{name}.png",
        )

        results.append(
            {
                "model": name,
                "roc_auc": float(roc),
                "macro_f1": float(macro_f1),
                "minority_precision_rejected": float(precision_min),
                "minority_recall_rejected": float(recall_min),
                "minority_f1_rejected": float(f1_min),
            }
        )

        if f1_min > best_f1_min:
            best_f1_min = f1_min
            best = (name, model)

        # Feature importance plots for tree-based models
        if name in {"random_forest", "decision_tree", "xgboost"}:
            try:
                importances = model.feature_importances_
                # Sort and plot top 20
                idx = np.argsort(importances)[::-1][:20]
                plt.figure(figsize=(8, 5))
                plt.bar(range(len(idx)), importances[idx])
                plt.xticks(range(len(idx)), [str(i) for i in idx], rotation=45, ha="right")
                plt.title(f"Top feature importances ({name}) - by transformed feature index")
                plt.tight_layout()
                plt.savefig(cfg.figs_dir / f"feature_importances_{name}.png")
                plt.close()
            except Exception:
                pass

    # Save best model
    assert best is not None
    best_name, best_model = best
    best_model_path = cfg.models_dir / "best_model.pkl"
    joblib.dump(best_model, best_model_path)

    # Save comparison table
    comp_df = pd.DataFrame(results).sort_values(
        by="minority_f1_rejected", ascending=False
    )
    comp_df.to_csv(cfg.models_dir / "model_comparison.csv", index=False)

    # Bar chart comparison by minority F1
    plt.figure(figsize=(8, 4))
    sns.barplot(data=comp_df, x="model", y="minority_f1_rejected")
    plt.title("Model comparison (Rejected minority class F1)")
    plt.tight_layout()
    plt.savefig(cfg.figs_dir / "model_comparison_bar.png")
    plt.close()

    # ROC curves
    plt.figure(figsize=(7, 5))
    from sklearn.metrics import roc_curve

    for name, model in models.items():
        prob = model.predict_proba(X_test_t)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        plt.plot(fpr, tpr, label=name)

    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(cfg.figs_dir / "roc_curve_comparison.png")
    plt.close()


    print("Done.")
    print(f"Best model: {best_name} with minority_f1_rejected={best_f1_min}")


if __name__ == "__main__":
    main()


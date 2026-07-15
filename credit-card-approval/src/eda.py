"""EDA utilities for the credit card approval dataset.

Run (after creating venv + installing requirements):
  python -m src.eda

This script downloads the dataset, merges records, engineers AGE_YEARS/EMPLOYED_YEARS,
then generates basic plots into notebooks/figures/.

Note: training script already creates preprocessing + evaluation artifacts.
"""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


DATA_URL = (
    "https://raw.githubusercontent.com/bmariasirine/Credit_Card_Approval_Prediction/main/datasets.zip"
)


def download_if_needed(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "datasets.zip"
    app_path = out_dir / "application_record.csv"
    cred_path = out_dir / "credit_record.csv"
    if not (zip_path.exists() and app_path.exists() and cred_path.exists()):
        if not zip_path.exists():
            print(f"Downloading dataset -> {zip_path}")
            urllib.request.urlretrieve(DATA_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)
    return out_dir


def main():
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data"
    figs_dir = base_dir / "notebooks" / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    download_if_needed(data_dir)
    app = pd.read_csv(data_dir / "application_record.csv")
    cred = pd.read_csv(data_dir / "credit_record.csv")

    high_risk_status = {2, 3, 4, 5}
    cred_flag = cred.assign(high_risk=cred["STATUS"].isin(high_risk_status).astype(int))
    target_df = cred_flag.groupby("ID", as_index=False)["high_risk"].max()
    target_df["TARGET"] = 1 - target_df["high_risk"]
    target_df = target_df[["ID", "TARGET"]]

    df = app.merge(target_df, on="ID", how="inner")

    df["AGE_YEARS"] = (df["DAYS_BIRTH"].clip(lower=0) / 365.25).astype(float)
    sentinel = 365243
    df["EMPLOYED_YEARS"] = df["DAYS_EMPLOYED"].apply(lambda x: 0.0 if x == sentinel else float(abs(x) / 365.25))

    # class balance
    plt.figure(figsize=(6, 4))
    df["TARGET"].value_counts(normalize=True).plot(kind="bar")
    plt.title("Target Class Balance")
    plt.xticks([0, 1], ["Rejected(0)", "Approved(1)"], rotation=0)
    plt.tight_layout()
    plt.savefig(figs_dir / "target_balance.png")
    plt.close()

    # categorical count plots
    cat_cols = [
        "CODE_GENDER",
        "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY",
        "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS",
        "NAME_HOUSING_TYPE",
        "OCCUPATION_TYPE",
    ]

    for c in cat_cols:
        if c not in df.columns:
            continue
        plt.figure(figsize=(10, 4))
        sns.countplot(data=df, x=c, hue="TARGET")
        plt.title(f"{c} vs TARGET")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(figs_dir / f"count_{c}_vs_target.png")
        plt.close()

    # distribution plots
    for c in ["AMT_INCOME_TOTAL", "AGE_YEARS", "EMPLOYED_YEARS", "CNT_CHILDREN", "CNT_FAM_MEMBERS"]:
        if c not in df.columns:
            continue
        plt.figure(figsize=(7, 4))
        sns.histplot(data=df, x=c, hue="TARGET", bins=30, element="step", stat="density", common_norm=False)
        plt.title(f"Distribution of {c} by TARGET")
        plt.tight_layout()
        plt.savefig(figs_dir / f"dist_{c}_by_target.png")
        plt.close()

    # correlation heatmap numeric subset (note: includes engineered cols)
    numeric_candidates = [
        "CNT_CHILDREN",
        "AMT_INCOME_TOTAL",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "CNT_FAM_MEMBERS",
        "AGE_YEARS",
        "EMPLOYED_YEARS",
    ]
    numeric_cols = [c for c in numeric_candidates if c in df.columns]
    corr = df[numeric_cols].corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Heatmap (numeric)")
    plt.tight_layout()
    plt.savefig(figs_dir / "correlation_heatmap.png")
    plt.close()

    print("EDA plots saved to notebooks/figures/")


if __name__ == "__main__":
    main()


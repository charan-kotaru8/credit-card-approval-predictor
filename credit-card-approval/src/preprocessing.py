"""Preprocessing pipeline builder.

This module is mainly for clarity and reuse. The training script currently
implements a full end-to-end preprocessing fit.

You can extend this to export the fitted preprocessor for web scoring.
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class PreprocessConfig:
    numeric_features: list[str]
    binary_features: list[str]
    categorical_features: list[str]


def build_preprocessor(cfg: PreprocessConfig) -> ColumnTransformer:
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

    binary_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, cfg.numeric_features),
            ("bin", binary_transformer, cfg.binary_features),
            ("cat", categorical_transformer, cfg.categorical_features),
        ],
        remainder="drop",
    )


def save_preprocessor(preprocessor, out_path: str) -> None:
    joblib.dump(preprocessor, out_path)


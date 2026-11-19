"""Preprocessing pipeline (no data leakage).

- Target `Sales_Category` is derived from `Global_Sales` only for labeling.
- Input features: Platform, Genre, Publisher, Year (categorical/numeric). Sales columns are excluded.
- Encoding is fit on training data only via sklearn Pipeline/ColumnTransformer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .data_loader import FEATURE_COLUMNS, TARGET_COLUMN

SALES_BINS = [-np.inf, 0.5, 2.0, np.inf]
SALES_LABELS = ["Rendah", "Sedang", "Tinggi"]
SALES_RULE = {
    "Rendah": "Global_Sales < 0.5 (juta kopi)",
    "Sedang": "0.5 <= Global_Sales <= 2.0 (juta kopi)",
    "Tinggi": "Global_Sales > 2.0 (juta kopi)",
}


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    if "Global_Sales" not in df.columns:
        raise ValueError("Kolom Global_Sales diperlukan untuk membentuk target.")
    out = df.copy()
    sales = pd.to_numeric(out["Global_Sales"], errors="coerce")
    out[TARGET_COLUMN] = pd.cut(
        sales,
        bins=SALES_BINS,
        labels=SALES_LABELS,
        include_lowest=True,
    )
    out.loc[sales.eq(0.5), TARGET_COLUMN] = "Sedang"
    return out


def clean_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop rows that cannot be used reliably; return cleaned df and report."""
    before = len(df)
    out = df.copy()
    # Year must be numeric for the model; Publisher missing handled by imputer but Year missing is dropped
    # to avoid leaking central tendency choices in opaque ways at row level? We keep rows and impute Year.
    out["Year"] = pd.to_numeric(out["Year"], errors="coerce")
    report = {
        "rows_before": before,
        "rows_after": int(len(out)),
        "year_missing": int(out["Year"].isnull().sum()),
        "publisher_missing": int(out["Publisher"].isnull().sum()) if "Publisher" in out else 0,
        "target_missing": int(out[TARGET_COLUMN].isnull().sum()) if TARGET_COLUMN in out else 0,
    }
    out = out[out[TARGET_COLUMN].notna()].copy()
    report["rows_after_target_drop"] = int(len(out))
    report["dropped_rows"] = before - report["rows_after_target_drop"]
    return out, report


def build_preprocessor() -> ColumnTransformer:
    categorical = ["Platform", "Genre", "Publisher"]
    numeric = ["Year"]

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, categorical),
            ("num", num_pipe, numeric),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def split_features_target(df: pd.DataFrame):
    missing_feats = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_feats:
        raise ValueError(f"Fitur tidak lengkap: {missing_feats}")
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target {TARGET_COLUMN} belum dibuat.")
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(str)
    return X, y


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Extract feature names after fit."""
    try:
        names = preprocessor.get_feature_names_out()
        return [str(n) for n in names]
    except Exception:
        names: list[str] = []
        cat = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        names.extend([str(n) for n in cat.get_feature_names_out(["Platform", "Genre", "Publisher"])])
        names.extend(["Year"])
        return names

"""Load and audit the Kaggle VGChartz dataset."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = {
    "Rank",
    "Name",
    "Platform",
    "Year",
    "Genre",
    "Publisher",
    "NA_Sales",
    "EU_Sales",
    "JP_Sales",
    "Other_Sales",
    "Global_Sales",
}

# Features used for prediction (no leakage: no sales columns, no Global_Sales-derived target input).
FEATURE_COLUMNS = ["Platform", "Genre", "Publisher", "Year"]
TARGET_COLUMN = "Sales_Category"


@dataclass
class AuditResult:
    shape: tuple[int, int]
    columns: list[str]
    dtypes: dict[str, str]
    missing: dict[str, int]
    duplicates: int
    head_rows: list[dict[str, Any]]
    year_min: float | None
    year_max: float | None


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {sorted(missing)}")
    return df


def audit_dataset(df: pd.DataFrame) -> AuditResult:
    year = pd.to_numeric(df["Year"], errors="coerce")
    return AuditResult(
        shape=(int(df.shape[0]), int(df.shape[1])),
        columns=list(df.columns),
        dtypes={c: str(t) for c, t in df.dtypes.items()},
        missing={c: int(v) for c, v in df.isnull().sum().items()},
        duplicates=int(df.duplicated().sum()),
        head_rows=df.head(5).to_dict(orient="records"),
        year_min=float(year.min()) if year.notna().any() else None,
        year_max=float(year.max()) if year.notna().any() else None,
    )

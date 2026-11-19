"""Application services: load saved pipeline, predict, and serve dashboard data."""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

# Ensure project root importable
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data_loader import FEATURE_COLUMNS, load_dataset  # noqa: E402
from src.preprocessing import SALES_RULE, create_target  # noqa: E402

LABELS = ["Rendah", "Sedang", "Tinggi"]
logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    prediction: str
    confidence: float
    probabilities: dict[str, float]
    input_data: dict[str, Any]
    disclaimer: str


class ModelService:
    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.models_dir = self.base_dir / "data" / "models"
        self.pipeline = None
        self.metadata: dict[str, Any] = {}
        self.load_error: str | None = None
        self._df_cache: pd.DataFrame | None = None
        self._dataset_load_attempted = False
        self.dataset_error: str | None = None
        self.load()

    @property
    def pipeline_path(self) -> Path:
        return self.models_dir / "decision_tree_pipeline.joblib"

    @property
    def metadata_path(self) -> Path:
        return self.models_dir / "model_metadata.json"

    def load(self) -> None:
        self.load_error = None
        try:
            if self.metadata_path.exists():
                self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except Exception:
            self.metadata = {}
            logger.exception("Failed to load model metadata from %s", self.metadata_path)
            self.load_error = "Gagal memuat metadata model. Periksa log aplikasi."
        try:
            if not self.pipeline_path.exists():
                self.load_error = (
                    f"Model belum tersedia: {self.pipeline_path.name} tidak ditemukan. "
                    "Jalankan `python train_model.py` terlebih dahulu."
                )
                logger.error("Model pipeline not found at %s", self.pipeline_path)
                self.pipeline = None
                return
            self.pipeline = joblib.load(self.pipeline_path)
        except Exception:
            self.pipeline = None
            logger.exception("Failed to load model pipeline from %s", self.pipeline_path)
            self.load_error = "Gagal memuat pipeline model. Periksa log aplikasi."

    def is_ready(self) -> bool:
        return self.pipeline is not None

    def is_dataset_ready(self) -> bool:
        return self._get_dataset() is not None

    def get_allowed_options(self) -> dict[str, list[str]]:
        # Prefer metadata/audit knowledge; fall back to dataset.
        df = self._get_dataset()
        if df is None:
            return {"Platform": [], "Genre": [], "Publisher": []}
        return {
            "Platform": sorted([str(x) for x in df["Platform"].dropna().unique()]),
            "Genre": sorted([str(x) for x in df["Genre"].dropna().unique()]),
            "Publisher": sorted([str(x) for x in df["Publisher"].dropna().unique()]),
        }

    def validate_input(self, data: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
        errors: list[str] = []
        cleaned: dict[str, Any] = {}
        required = ["Platform", "Genre", "Publisher", "Year"]
        for f in required:
            if f not in data or str(data.get(f, "")).strip() == "":
                errors.append(f"Field '{f}' wajib diisi.")
        if errors:
            return None, errors

        cleaned["Platform"] = str(data["Platform"]).strip()
        cleaned["Genre"] = str(data["Genre"]).strip()
        cleaned["Publisher"] = str(data["Publisher"]).strip()
        try:
            year_val = float(data["Year"])
            if not pd.notna(year_val) or not year_val.is_integer():
                raise ValueError
            year_int = int(year_val)
            if year_int < 1970 or year_int > 2035:
                errors.append("Year harus berada pada rentang wajar (1970–2035).")
            cleaned["Year"] = year_int
        except Exception:
            errors.append("Year harus berupa bilangan bulat yang valid.")

        if errors:
            return None, errors
        # Note: unknown categories are handled by OneHotEncoder(handle_unknown='ignore') in pipeline.
        return cleaned, []

    def predict(self, data: dict[str, Any]) -> PredictionResult:
        if not self.is_ready():
            raise RuntimeError(self.load_error or "Model belum dimuat.")
        cleaned, errors = self.validate_input(data)
        if errors:
            raise ValueError(" | ".join(errors))
        X = pd.DataFrame([{k: cleaned[k] for k in FEATURE_COLUMNS}])
        pred = self.pipeline.predict(X)[0]
        probs = {}
        if hasattr(self.pipeline, "predict_proba"):
            proba = self.pipeline.predict_proba(X)[0]
            classes = list(getattr(self.pipeline, "classes_", LABELS))
            probs = {str(c): round(float(p), 6) for c, p in zip(classes, proba)}
        # Ensure all labels present
        for lbl in LABELS:
            probs.setdefault(lbl, 0.0)
        probs = {k: probs[k] for k in LABELS}
        confidence = max(probs.values()) if probs else 0.0
        return PredictionResult(
            prediction=str(pred),
            confidence=float(confidence),
            probabilities=probs,
            input_data=cleaned,
            disclaimer="Hasil ini merupakan estimasi model berdasarkan data historis, bukan kepastian penjualan.",
        )

    def _get_dataset(self) -> pd.DataFrame | None:
        if self._df_cache is not None:
            return self._df_cache
        if self._dataset_load_attempted:
            return None
        self._dataset_load_attempted = True
        path = self.base_dir / "vgsales.csv"
        if not path.exists():
            self.dataset_error = f"Dataset tidak ditemukan: {path.name}. Letakkan file dataset pada folder proyek."
            logger.error("Dataset not found at %s", path)
            return None
        try:
            df = load_dataset(path)
            df = create_target(df)
            self._df_cache = df
            return df
        except Exception:
            logger.exception("Failed to load dataset from %s", path)
            self.dataset_error = "Gagal memuat dataset. Periksa format dan kolom wajib, lalu lihat log aplikasi."
            return None

    def get_dashboard_data(self) -> dict[str, Any]:
        df = self._get_dataset()
        meta = self.metadata or {}
        results = meta.get("results", {})
        final_test = results.get("final", {}).get("test", {})
        if df is None:
            return {
                "error": self.dataset_error or "Dataset tidak tersedia.",
                "dataset_error": self.dataset_error or "Dataset tidak tersedia.",
                "metadata": meta,
                "final_metrics": final_test,
                "model_ready": self.is_ready(),
                "load_error": self.load_error,
            }
        audit = meta.get("audit", {})
        dist = df["Sales_Category"].value_counts()
        by_genre = df["Genre"].value_counts().head(12).to_dict()
        by_platform = df["Platform"].value_counts().head(12).to_dict()
        year_counts = df.dropna(subset=["Year"]).assign(Year=pd.to_numeric(df["Year"], errors="coerce")).dropna(subset=["Year"])
        year_counts["Year"] = year_counts["Year"].astype(int)
        by_year = year_counts["Year"].value_counts().sort_index().to_dict()
        cat_genre = (
            df.groupby(["Genre", "Sales_Category"], observed=True).size().unstack(fill_value=0)
        )
        cat_platform = (
            df.groupby(["Platform", "Sales_Category"], observed=True).size().unstack(fill_value=0)
        )
        top_platforms = df["Platform"].value_counts().head(10).index.tolist()
        cat_platform = cat_platform.loc[[p for p in top_platforms if p in cat_platform.index]]
        return {
            "metadata": meta,
            "audit": audit,
            "total_rows": int(len(df)),
            "duplicates": int(df.duplicated().sum()),
            "missing": {c: int(v) for c, v in df.isnull().sum().items()},
            "target_dist": {k: int(dist.get(k, 0)) for k in LABELS},
            "target_pct": {k: round(100 * int(dist.get(k, 0)) / len(df), 2) for k in LABELS},
            "by_genre": by_genre,
            "by_platform": by_platform,
            "by_year": {str(k): int(v) for k, v in by_year.items()},
            "cat_genre": {g: {c: int(row.get(c, 0)) for c in LABELS} for g, row in cat_genre.to_dict(orient="index").items()},
            "cat_platform": {p: {c: int(row.get(c, 0)) for c in LABELS} for p, row in cat_platform.to_dict(orient="index").items()},
            "final_metrics": final_test,
            "sales_rules": SALES_RULE,
            "n_genres": int(df["Genre"].nunique()),
            "n_platforms": int(df["Platform"].nunique()),
            "n_publishers": int(df["Publisher"].nunique()),
            "model_ready": self.is_ready(),
            "load_error": self.load_error,
        }

    def get_dataset_page(self, page: int = 1, per_page: int = 25):
        df = self._get_dataset()
        if df is None:
            return {
                "rows": [],
                "page": 1,
                "total_pages": 1,
                "total": 0,
                "error": self.dataset_error or "Dataset tidak tersedia.",
            }
        total = len(df)
        per_page = max(5, min(100, per_page))
        total_pages = max(1, (total + per_page - 1) // per_page)
        page = min(max(1, page), total_pages)
        start = (page - 1) * per_page
        end = start + per_page
        chunk = df.iloc[start:end].copy()
        # Replace NaN for JSON/template safety
        chunk = chunk.where(pd.notnull(chunk), None)
        rows = chunk.to_dict(orient="records")
        return {"rows": rows, "page": page, "total_pages": total_pages, "total": total, "per_page": per_page}

    def get_feature_importance(self):
        if not self.is_ready():
            return []
        try:
            model = self.pipeline.named_steps["model"]
            pre = self.pipeline.named_steps["preprocess"]
            names = list(pre.get_feature_names_out())
            imps = getattr(model, "feature_importances_", None)
            if imps is None:
                return []
            pairs = sorted(zip(names, imps), key=lambda x: x[1], reverse=True)
            return [{"feature": n, "importance": float(v)} for n, v in pairs[:20] if float(v) > 0]
        except Exception:
            return []

"""Routes for the Game Sales Prediction dashboard."""
from __future__ import annotations

import logging
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file

from .services import LABELS, ModelService

logger = logging.getLogger(__name__)


def register_routes(app: Flask) -> None:
    base_dir = Path(app.config["BASE_DIR"])
    svc = ModelService(base_dir)
    app.extensions["model_service"] = svc

    @app.context_processor
    def inject_globals():
        return {
            "identity": {"name": "Adi Hadianto", "nim": "301240006", "class": "5B"},
            "labels": LABELS,
            "model_ready": svc.is_ready(),
        }

    @app.errorhandler(404)
    def not_found(_e):
        return render_template("error.html", code=404, message="Halaman tidak ditemukan."), 404

    @app.errorhandler(500)
    def server_error(_e):
        return render_template("error.html", code=500, message="Terjadi kesalahan pada server."), 500

    @app.get("/")
    @app.get("/dashboard")
    def dashboard():
        data = svc.get_dashboard_data()
        # chart payloads (JSON-safe)
        charts = {
            "target": data.get("target_dist", {}),
            "by_genre": data.get("by_genre", {}),
            "by_platform": data.get("by_platform", {}),
            "by_year": data.get("by_year", {}),
        }
        return render_template("dashboard.html", data=data, charts=charts)

    @app.route("/predict", methods=["GET", "POST"])
    def predict():
        options = svc.get_allowed_options()
        result = None
        errors = []
        form = {"Platform": "", "Genre": "", "Publisher": "", "Year": ""}
        if request.method == "POST":
            form = {
                "Platform": request.form.get("Platform", ""),
                "Genre": request.form.get("Genre", ""),
                "Publisher": request.form.get("Publisher", ""),
                "Year": request.form.get("Year", ""),
            }
            try:
                r = svc.predict(form)
                result = {
                    "prediction": r.prediction,
                    "confidence": r.confidence,
                    "probabilities": r.probabilities,
                    "input_data": r.input_data,
                    "disclaimer": r.disclaimer,
                }
            except ValueError as e:
                errors = [str(e)]
            except Exception:
                logger.exception("Prediction request failed")
                errors = ["Prediksi gagal diproses. Silakan coba kembali atau periksa log aplikasi."]
        return render_template(
            "predict.html",
            options=options,
            result=result,
            errors=errors,
            form=form,
            load_error=svc.load_error,
            dataset_error=svc.dataset_error,
        )

    @app.get("/dataset")
    def dataset():
        page = request.args.get("page", 1, type=int)
        pager = svc.get_dataset_page(page=page, per_page=25)
        data = svc.get_dashboard_data()
        charts = {
            "target": data.get("target_dist", {}),
            "by_genre": data.get("by_genre", {}),
            "by_platform": data.get("by_platform", {}),
            "by_year": data.get("by_year", {}),
            "cat_genre": data.get("cat_genre", {}),
            "cat_platform": data.get("cat_platform", {}),
        }
        return render_template("dataset.html", pager=pager, data=data, charts=charts)

    @app.get("/evaluation")
    def evaluation():
        meta = svc.metadata or {}
        return render_template("evaluation.html", meta=meta, load_error=svc.load_error)

    @app.get("/tree")
    def tree():
        meta = svc.metadata or {}
        importance = svc.get_feature_importance()
        img_path = base_dir / "data" / "models" / "decision_tree_viz.png"
        static_img = base_dir / "app" / "static" / "images" / "decision_tree_viz.png"
        tree_image_available = img_path.exists() or static_img.exists()
        return render_template(
            "tree.html",
            meta=meta,
            importance=importance,
            tree_image=tree_image_available,
        )

    @app.get("/tree/image")
    def tree_image():
        img_path = base_dir / "data" / "models" / "decision_tree_viz.png"
        static_img = base_dir / "app" / "static" / "images" / "decision_tree_viz.png"
        image_path = img_path if img_path.exists() else static_img
        if not image_path.exists():
            abort(404)
        return send_file(
            image_path,
            mimetype="image/png",
            download_name="decision_tree_viz.png",
            conditional=True,
        )

    @app.get("/about")
    def about():
        meta = svc.metadata or {}
        return render_template("about.html", meta=meta)

    @app.get("/api/health")
    def health():
        dataset_ready = svc.is_dataset_ready()
        return jsonify({
            "status": "ok" if svc.is_ready() and dataset_ready else "degraded",
            "model_ready": svc.is_ready(),
            "dataset_ready": dataset_ready,
            "load_error": svc.load_error,
            "dataset_error": svc.dataset_error,
        })

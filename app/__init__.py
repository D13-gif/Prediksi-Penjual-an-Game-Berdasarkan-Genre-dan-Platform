"""Application factory."""
from __future__ import annotations

import os
import secrets
from pathlib import Path

from flask import Flask


def create_app(base_dir: str | Path | None = None) -> Flask:
    base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parents[1]
    app = Flask(
        __name__,
        template_folder=str(base_dir / "app" / "templates"),
        static_folder=str(base_dir / "app" / "static"),
        static_url_path="/static",
    )
    app.config["BASE_DIR"] = str(base_dir)
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
    app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024

    @app.template_filter("pct")
    def pct(v, digits=1):
        try:
            return f"{float(v) * 100:.{digits}f}%"
        except Exception:
            return "-"

    @app.template_filter("fmt4")
    def fmt4(v):
        try:
            return f"{float(v):.4f}"
        except Exception:
            return str(v)

    @app.template_filter("comma")
    def comma(v):
        try:
            return f"{int(v):,}"
        except Exception:
            return str(v)

    from .routes import register_routes

    register_routes(app)
    return app

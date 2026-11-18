"""Launch the local FocalBench HTTP application."""

import os
from pathlib import Path

from flask import Flask, request, send_from_directory
from waitress import serve
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from .models import WorkbenchError
from .reconstruction import ALGORITHM_VERSION, Reconstruction
from .results_api import results
from .stacks_api import stacks
from .store import Library

STATIC = Path(__file__).resolve().parent.parent / "static"
REQUEST_LIMIT = 105 * 1024 * 1024


def create_app(data_folder: Path | str | None = None):
    app = Flask(__name__, static_folder=STATIC, static_url_path="/static")
    app.config.update(
        MAX_CONTENT_LENGTH=REQUEST_LIMIT,
        MAX_FORM_MEMORY_SIZE=600_000,
        MAX_FORM_PARTS=96,
        TRUSTED_HOSTS=["localhost", "127.0.0.1", "[::1]"],
    )
    root = Path(data_folder or os.getenv("FOCALBENCH_DATA_DIR", "runtime-data"))
    library = Library(root)
    app.extensions["focalbench"] = {"library": library, "engine": Reconstruction(library)}
    app.register_blueprint(stacks)
    app.register_blueprint(results)

    @app.get("/")
    def workbench():
        return send_from_directory(STATIC, "index.html", max_age=0)

    @app.get("/api/health")
    def health():
        return {"app": "FocalBench", "status": "ok", "version": ALGORITHM_VERSION, **library.check()}

    @app.errorhandler(WorkbenchError)
    def workbench_problem(error):
        return error.public(), error.status

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(error):
        return {"error": "The upload exceeds the local request limit."}, 413

    @app.errorhandler(HTTPException)
    def framework_problem(error):
        if not request.path.startswith("/api/"):
            return error
        return {"error": error.description}, error.code

    @app.after_request
    def common_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        if request.path == "/" or request.path.startswith("/api/"):
            response.headers.setdefault("Referrer-Policy", "same-origin")
        if request.path.startswith("/api/") and response.is_json:
            response.headers["Cache-Control"] = "no-store"
        return response

    return app


def main():
    host = os.getenv("FOCALBENCH_HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "18944"))
    application = create_app()
    serve(
        application, host=host, port=port, threads=5, max_request_body_size=REQUEST_LIMIT,
        expose_tracebacks=False, channel_timeout=75,
    )

"""MediBot – Health check utility & route."""

import datetime
import os
import sys

from flask import Blueprint, jsonify

health_bp = Blueprint("health", __name__)

_start_time = datetime.datetime.utcnow()


@health_bp.get("/health")
def health():
    """
    Health check endpoint.
    Returns model status, DB reachability, and uptime.
    Used by Render, Docker HEALTHCHECK, and monitoring tools.
    """
    status = {"status": "ok", "uptime_seconds": _uptime(), "timestamp": _now()}

    # Model status
    try:
        from app.services.embedder import EmbedderClassifier
        clf = EmbedderClassifier.get()
        status["model"] = {
            "type": "sentence-transformer",
            "ready": clf.available,
        }
    except Exception as exc:
        try:
            import chat  # legacy
            status["model"] = {"type": "bilstm-legacy", "ready": True}
        except Exception:
            status["model"] = {"type": "unavailable", "ready": False}
            status["status"] = "degraded"

    # Database status
    try:
        from database import count_users
        count_users()
        status["database"] = {"ready": True}
    except Exception as exc:
        status["database"] = {"ready": False, "error": str(exc)}
        status["status"] = "degraded"

    # Severity classifier
    try:
        from app.services.severity import SeverityClassifier
        status["severity_classifier"] = {"ml": SeverityClassifier.get().available}
    except Exception:
        status["severity_classifier"] = {"ml": False}

    http_code = 200 if status["status"] == "ok" else 503
    return jsonify(status), http_code


def _uptime() -> int:
    return int((datetime.datetime.utcnow() - _start_time).total_seconds())


def _now() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"

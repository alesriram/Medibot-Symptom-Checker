"""
MediBot AI – Flask Application Factory (Render Optimized)
"""

import os
import logging

# ── REQUIRED IMPORTS ─────────────────────────────────────
from flask import Flask, render_template, session, redirect, url_for
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from database import init_db

# ── NLTK setup ───────────────────────────────────────────
def _ensure_nltk():
    import nltk
    packages = [
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("punkt",     "tokenizers/punkt"),
        ("wordnet",   "corpora/wordnet"),
    ]
    for pkg, res in packages:
        try:
            nltk.data.find(res)
        except (LookupError, OSError):
            nltk.download(pkg)
# ── Logging ─────────────────────────────────────────────
try:
    from pythonjsonlogger import jsonlogger

    handler = logging.StreamHandler()
    handler.setFormatter(jsonlogger.JsonFormatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s"
    ))

    logging.basicConfig(level=logging.INFO, handlers=[handler])

except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

log = logging.getLogger(__name__)

# ── App factory ─────────────────────────────────────────
def create_app():
    app = Flask(__name__)
    app.secret_key = os.environ.get("MEDIBOT_SECRET_KEY", os.urandom(32))

    # ── CORS ─────────────────────────────────────────────
    CORS(app, supports_credentials=True)

    # ── Rate limiter ─────────────────────────────────────
    limiter = Limiter(
        key_func=get_remote_address,
        app=app,
        default_limits=[],
        storage_uri="memory://",
    )

    # ── Blueprints ───────────────────────────────────────
    from app.routes.auth import auth_bp
    from app.routes.admin import admin_bp
    from app.routes.chat import chat_bp
    from app.utils.health import health_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(health_bp)

    # Apply rate limit to chat routes
    limiter.limit("30 per minute")(chat_bp)
    app.register_blueprint(chat_bp)

    # ── Routes ───────────────────────────────────────────
    @app.get("/")
    def index():
        return render_template("base.html")

    @app.route("/login")
    def login():
        return render_template("login.html")

    @app.route("/admin")
    def admin():
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return render_template(
            "admin.html",
            username=session.get("username", "Admin")
        )

    @app.route("/about")
    def about_us():
        return render_template("about_us.html")

    # ── Startup tasks (SAFE) ─────────────────────────────
    with app.app_context():
        _ensure_nltk()
        init_db()

    return app


# ── Entry point ─────────────────────────────────────────
app = create_app()

if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug_mode)
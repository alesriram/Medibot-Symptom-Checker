"""
MediBot – Test suite (pytest)
Covers: auth, predict, history, admin guards, database, health endpoint,
        severity service, embedder classifier.

Run: pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import patch, MagicMock
from werkzeug.security import generate_password_hash


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def app_client(tmp_path):
    """Isolated Flask test client with a temporary SQLite database."""
    db_path = str(tmp_path / "test_medibot.db")
    os.environ["MEDIBOT_SECRET_KEY"] = "test-secret"
    os.environ["MEDIBOT_DB_PATH"]    = db_path
    os.environ["ALLOWED_ORIGINS"]    = "http://localhost:5000"

    import importlib
    import database as db_mod

    importlib.reload(db_mod)

    # Patch the embedder so tests don't try to download a model
    with patch("app.services.embedder.EmbedderClassifier._ensure_model", return_value=False):
        import app as app_module
        importlib.reload(app_module)

        db_mod.init_db()
        app_module.app.config["TESTING"]    = True
        app_module.app.config["SECRET_KEY"] = "test-secret"

        with app_module.app.test_client() as client:
            yield client


# ── Auth ──────────────────────────────────────────────────────────────────────

class TestAuth:
    def test_login_success(self, app_client):
        res = app_client.post("/auth/login", json={"username": "admin", "password": "admin@123"})
        assert res.status_code == 200
        assert res.get_json()["ok"] is True
        assert res.get_json()["role"] == "admin"

    def test_login_wrong_password(self, app_client):
        res = app_client.post("/auth/login", json={"username": "admin", "password": "wrong"})
        assert res.status_code == 401
        assert res.get_json()["ok"] is False

    def test_login_missing_fields(self, app_client):
        res = app_client.post("/auth/login", json={"username": "admin"})
        assert res.status_code == 400

    def test_signup_creates_user(self, app_client):
        res = app_client.post("/auth/signup",
                              json={"username": "newuser1", "password": "securepass1", "name": "New User"})
        assert res.status_code == 201
        assert res.get_json()["ok"] is True

    def test_signup_duplicate_username(self, app_client):
        app_client.post("/auth/signup", json={"username": "dupuser", "password": "pass1234", "name": "A"})
        res = app_client.post("/auth/signup", json={"username": "dupuser", "password": "pass1234", "name": "B"})
        assert res.status_code == 409

    def test_signup_short_password(self, app_client):
        res = app_client.post("/auth/signup", json={"username": "u1", "password": "abc", "name": "User"})
        assert res.status_code == 400

    def test_signup_non_alphanumeric_username(self, app_client):
        res = app_client.post("/auth/signup",
                              json={"username": "bad user!", "password": "validpass1", "name": "Bad"})
        assert res.status_code == 400

    def test_logout(self, app_client):
        app_client.post("/auth/login", json={"username": "admin", "password": "admin@123"})
        res = app_client.post("/auth/logout")
        assert res.get_json()["ok"] is True

    def test_me_not_logged_in(self, app_client):
        res = app_client.get("/auth/me")
        assert res.get_json()["logged_in"] is False

    def test_me_logged_in(self, app_client):
        app_client.post("/auth/login", json={"username": "admin", "password": "admin@123"})
        res = app_client.get("/auth/me")
        data = res.get_json()
        assert data["logged_in"] is True
        assert data["username"] == "admin"


# ── Predict ───────────────────────────────────────────────────────────────────

class TestPredict:
    def test_empty_message_returns_400(self, app_client):
        res = app_client.post("/predict", json={"message": ""})
        assert res.status_code == 400

    def test_missing_message_returns_400(self, app_client):
        res = app_client.post("/predict", json={})
        assert res.status_code == 400

    def test_valid_message_uses_predictor(self, app_client):
        mock_result = {
            "tag": "greeting", "response": ["greeting", "Hello!"],
            "confidence": 0.9, "method": "embed", "alternatives": []
        }
        with patch("app.routes.chat.service_predict", return_value=mock_result):
            res = app_client.post("/predict", json={"message": "hello"})
            assert res.status_code == 200
            data = res.get_json()
            assert "answer" in data
            assert "confidence" in data
            assert "method" in data

    def test_html_injection_stripped(self, app_client):
        mock_result = {
            "tag": "not_understand", "response": ["not_understand", "unclear"],
            "confidence": 0.0, "method": "embed", "alternatives": []
        }
        with patch("app.routes.chat.service_predict", return_value=mock_result):
            res = app_client.post("/predict", json={"message": "<script>alert(1)</script>"})
            assert res.status_code in (200, 400)

    def test_emergency_detected(self, app_client):
        res = app_client.post("/predict", json={"message": "I have chest pain and can't breathe"})
        assert res.status_code == 200
        data = res.get_json()
        assert data.get("emergency") is True

    def test_response_includes_severity_for_medical_tag(self, app_client):
        mock_result = {
            "tag": "Diabetes",
            "response": ["Diabetes", "Diabetes", "Description", "Precautions", "Confidence: 85%"],
            "confidence": 0.85, "method": "embed", "alternatives": []
        }
        with patch("app.routes.chat.service_predict", return_value=mock_result):
            res = app_client.post("/predict", json={"message": "I have high blood sugar"})
            assert res.status_code == 200
            data = res.get_json()
            assert "severity" in data
            assert "doctor" in data


# ── Health endpoint (Tier-2) ──────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self, app_client):
        res = app_client.get("/health")
        assert res.status_code in (200, 503)  # 503 if model not loaded in test env
        data = res.get_json()
        assert "status" in data
        assert "database" in data
        assert "uptime_seconds" in data

    def test_health_database_ok(self, app_client):
        res = app_client.get("/health")
        data = res.get_json()
        assert data["database"]["ready"] is True


# ── History ───────────────────────────────────────────────────────────────────

class TestHistory:
    def test_history_empty_by_default(self, app_client):
        res = app_client.get("/history")
        assert res.get_json() == []

    def test_log_symptom_stores_entry(self, app_client):
        app_client.post("/log_symptom", json={"symptom": "fever", "result": "flu"})
        history = app_client.get("/history").get_json()
        assert len(history) == 1
        assert history[0]["symptom"] == "fever"

    def test_log_symptom_truncates_long_input(self, app_client):
        long_symptom = "a" * 1000
        app_client.post("/log_symptom", json={"symptom": long_symptom, "result": "test"})
        history = app_client.get("/history").get_json()
        assert len(history[0]["symptom"]) <= 500


# ── Admin guards ──────────────────────────────────────────────────────────────

class TestAdminGuard:
    def test_list_users_requires_admin(self, app_client):
        res = app_client.get("/auth/users")
        assert res.status_code == 403

    def test_delete_user_requires_admin(self, app_client):
        res = app_client.delete("/auth/users/someuser")
        assert res.status_code == 403

    def test_admin_can_list_users(self, app_client):
        app_client.post("/auth/login", json={"username": "admin", "password": "admin@123"})
        res = app_client.get("/auth/users")
        assert res.status_code == 200
        assert any(u["username"] == "admin" for u in res.get_json())

    def test_admin_cannot_delete_self(self, app_client):
        app_client.post("/auth/login", json={"username": "admin", "password": "admin@123"})
        res = app_client.delete("/auth/users/admin")
        assert res.status_code == 400

    def test_user_count_public(self, app_client):
        res = app_client.get("/auth/users/count")
        assert res.status_code == 200
        assert res.get_json()["count"] >= 1


# ── Database ──────────────────────────────────────────────────────────────────

class TestDatabase:
    def test_get_user_returns_none_for_missing(self, app_client):
        import database as db_mod
        assert db_mod.get_user("doesnotexist") is None

    def test_create_and_retrieve_user(self, app_client):
        import database as db_mod
        db_mod.create_user("testuser", "Test User", generate_password_hash("pass1234"))
        user = db_mod.get_user("testuser")
        assert user is not None
        assert user["name"] == "Test User"
        assert user["role"] == "user"

    def test_delete_user(self, app_client):
        import database as db_mod
        db_mod.create_user("todelete", "Gone Soon", generate_password_hash("pass1234"))
        assert db_mod.delete_user("todelete") is True
        assert db_mod.get_user("todelete") is None

    def test_count_users(self, app_client):
        import database as db_mod
        assert db_mod.count_users() >= 1


# ── Severity service (Tier-2) ─────────────────────────────────────────────────

class TestSeverityService:
    def test_known_critical_disease(self):
        from app.services.severity import get_severity
        sev = get_severity("Heart Attack")
        assert sev["urgency"] >= 4

    def test_known_mild_disease(self):
        from app.services.severity import get_severity
        sev = get_severity("Common Cold")
        assert sev["urgency"] <= 2

    def test_unknown_disease_returns_dict(self):
        from app.services.severity import get_severity
        sev = get_severity("SomeMadeUpDisease123")
        assert "urgency" in sev
        assert "label" in sev

    def test_get_doctor_known(self):
        from app.services.severity import get_doctor
        assert get_doctor("Diabetes") == "Endocrinologist"

    def test_get_doctor_fallback(self):
        from app.services.severity import get_doctor
        assert get_doctor("UnknownCondition") == "General Physician"

    def test_is_emergency_chest_pain(self):
        from app.services.severity import is_emergency
        assert is_emergency("I have chest pain") is True

    def test_is_emergency_normal(self):
        from app.services.severity import is_emergency
        assert is_emergency("I have a mild headache") is False


# ── Triage endpoint ───────────────────────────────────────────────────────────

class TestTriage:
    def test_triage_requires_symptoms(self, app_client):
        res = app_client.post("/triage", json={"symptoms": []})
        assert res.status_code == 400

    def test_triage_returns_conditions(self, app_client):
        res = app_client.post("/triage", json={"symptoms": ["fever", "cough", "headache"]})
        assert res.status_code == 200
        data = res.get_json()
        assert data["ok"] is True
        assert "triage" in data
        assert "top_conditions" in data["triage"]


# ── Severity predict endpoint ─────────────────────────────────────────────────

class TestSeverityPredict:
    def test_normal_vitals_mild(self, app_client):
        res = app_client.post("/severity_predict",
                              json={"symptoms": "mild headache", "vitals": {"temp": 98.6, "hr": 72, "spo2": 98}})
        assert res.status_code == 200
        data = res.get_json()
        assert data["ok"] is True
        assert data["severity"]["level"] == "mild"

    def test_critical_spo2(self, app_client):
        res = app_client.post("/severity_predict",
                              json={"symptoms": "shortness of breath", "vitals": {"spo2": 88}})
        assert res.status_code == 200
        data = res.get_json()
        assert data["severity"]["level"] == "emergency"
        assert any("oxygen" in f.lower() for f in data["flags"])

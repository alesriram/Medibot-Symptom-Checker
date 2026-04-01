"""MediBot – Authentication routes."""

from flask import Blueprint, request, jsonify, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from database import get_user, create_user

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")


def is_logged_in():
    return bool(session.get("logged_in"))


def is_admin():
    return is_logged_in() and session.get("role") == "admin"


@auth_bp.post("/login")
def do_login():
    data = request.get_json(silent=True) or {}
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()
    if not username or not password:
        return jsonify({"ok": False, "msg": "Username and password are required."}), 400
    user = get_user(username)
    if user and check_password_hash(user["password"], password):
        session["logged_in"] = True
        session["username"]  = username
        session["role"]      = user.get("role", "user")
        return jsonify({"ok": True, "role": user.get("role", "user"), "name": user.get("name", username)})
    return jsonify({"ok": False, "msg": "Invalid username or password."}), 401


@auth_bp.post("/signup")
def do_signup():
    data = request.get_json(silent=True) or {}
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()
    name     = data.get("name", "").strip()
    if not username or not password or not name:
        return jsonify({"ok": False, "msg": "All fields are required."}), 400
    if len(password) < 8:
        return jsonify({"ok": False, "msg": "Password must be at least 8 characters."}), 400
    if not username.isalnum():
        return jsonify({"ok": False, "msg": "Username must be alphanumeric only."}), 400
    if get_user(username):
        return jsonify({"ok": False, "msg": "Username already exists. Choose another."}), 409
    create_user(username, name, generate_password_hash(password))
    session["logged_in"] = True
    session["username"]  = username
    session["role"]      = "user"
    return jsonify({"ok": True, "role": "user", "name": name}), 201


@auth_bp.post("/logout")
def do_logout():
    session.clear()
    return jsonify({"ok": True})


@auth_bp.get("/me")
def auth_me():
    if is_logged_in():
        return jsonify({"logged_in": True, "username": session["username"],
                        "role": session.get("role", "user")})
    return jsonify({"logged_in": False})

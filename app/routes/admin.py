"""MediBot – Admin routes."""

from flask import Blueprint, jsonify, session
from database import list_users, delete_user, count_users
from app.routes.auth import is_admin

admin_bp = Blueprint("admin", __name__, url_prefix="/auth")


@admin_bp.get("/users")
def list_users_route():
    if not is_admin():
        return jsonify({"ok": False, "msg": "Unauthorized"}), 403
    return jsonify(list_users())


@admin_bp.get("/users/count")
def count_users_route():
    return jsonify({"count": count_users()})


@admin_bp.delete("/users/<username>")
def delete_user_route(username):
    if not is_admin():
        return jsonify({"ok": False, "msg": "Unauthorized"}), 403
    if username == session.get("username"):
        return jsonify({"ok": False, "msg": "Cannot delete yourself."}), 400
    if not delete_user(username):
        return jsonify({"ok": False, "msg": "User not found."}), 404
    return jsonify({"ok": True, "msg": f"User '{username}' deleted."})

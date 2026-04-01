"""
MediBot – Database layer

Priority:
  1. MySQL  — set DB_HOST, DB_USER, DB_PASSWORD, DB_NAME env vars
              (use PlanetScale / Clever Cloud for free external MySQL)
  2. SQLite — fallback for local dev. DB stored at MEDIBOT_DB_PATH
              (default: /tmp/medibot.db so it never pollutes the repo)

MySQL Workbench setup:
  1. Create schema:  CREATE DATABASE medibot CHARACTER SET utf8mb4;
  2. Create user and grant permissions
  3. Set env vars in Render Dashboard -> Environment
"""

import os
import sqlite3
from contextlib import contextmanager
from werkzeug.security import generate_password_hash

USE_MYSQL = bool(os.environ.get("DB_HOST"))

if USE_MYSQL:
    import pymysql
    from pymysql.cursors import DictCursor

    DB_CONFIG = {
        "host":        os.environ["DB_HOST"],
        "user":        os.environ["DB_USER"],
        "password":    os.environ["DB_PASSWORD"],
        "database":    os.environ["DB_NAME"],
        "port":        int(os.environ.get("DB_PORT", 3306)),
        "cursorclass": DictCursor,
        "charset":     "utf8mb4",
        "connect_timeout": 10,
    }

    @contextmanager
    def db():
        conn = pymysql.connect(**DB_CONFIG)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_db():
        with db() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        username   VARCHAR(80)  PRIMARY KEY,
                        name       VARCHAR(120) NOT NULL,
                        password   VARCHAR(256) NOT NULL,
                        role       VARCHAR(20)  NOT NULL DEFAULT 'user',
                        created_at DATETIME     DEFAULT CURRENT_TIMESTAMP
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS chat_history (
                        id         INT AUTO_INCREMENT PRIMARY KEY,
                        username   VARCHAR(80),
                        symptom    TEXT,
                        result     VARCHAR(200),
                        severity   VARCHAR(20),
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_username (username)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """)
                cur.execute("SELECT COUNT(*) as cnt FROM users")
                if cur.fetchone()["cnt"] == 0:
                    cur.execute(
                        "INSERT INTO users (username,name,password,role) VALUES (%s,%s,%s,%s)",
                        ("admin", "Admin", generate_password_hash("admin@123"), "admin")
                    )

    def get_user(username):
        with db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM users WHERE username=%s", (username,))
                return cur.fetchone()

    def create_user(username, name, password_hash, role="user"):
        with db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO users (username,name,password,role) VALUES (%s,%s,%s,%s)",
                    (username, name, password_hash, role)
                )

    def delete_user(username):
        with db() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM users WHERE username=%s", (username,))
                return cur.rowcount > 0

    def list_users():
        with db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT username,name,role,created_at FROM users ORDER BY created_at")
                return cur.fetchall()

    def count_users():
        with db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) as cnt FROM users")
                return cur.fetchone()["cnt"]

    def save_chat(username, symptom, result, severity=""):
        with db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO chat_history (username,symptom,result,severity) VALUES (%s,%s,%s,%s)",
                    (username or "guest", symptom[:500], result[:200], severity)
                )

    def get_chat_history(username, limit=20):
        with db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM chat_history WHERE username=%s ORDER BY created_at DESC LIMIT %s",
                    (username, limit)
                )
                return cur.fetchall()

else:
    # ── SQLite fallback (local dev / Render free with no MySQL) ───────────────
    # Default to /tmp so the .db file is never created inside the repo
    DB_PATH = os.environ.get("MEDIBOT_DB_PATH", "/tmp/medibot.db")

    def _get_conn():
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @contextmanager
    def db():
        conn = _get_conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_db():
        with db() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    username   TEXT PRIMARY KEY,
                    name       TEXT NOT NULL,
                    password   TEXT NOT NULL,
                    role       TEXT NOT NULL DEFAULT 'user',
                    created_at TEXT DEFAULT (datetime('now'))
                );
                CREATE TABLE IF NOT EXISTS chat_history (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    username   TEXT,
                    symptom    TEXT,
                    result     TEXT,
                    severity   TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                );
            """)
            row = conn.execute("SELECT COUNT(*) FROM users").fetchone()
            if row[0] == 0:
                conn.execute(
                    "INSERT INTO users (username,name,password,role) VALUES (?,?,?,?)",
                    ("admin", "Admin", generate_password_hash("admin@123"), "admin")
                )

    def get_user(username):
        with db() as conn:
            row = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
            return dict(row) if row else None

    def create_user(username, name, password_hash, role="user"):
        with db() as conn:
            conn.execute(
                "INSERT INTO users (username,name,password,role) VALUES (?,?,?,?)",
                (username, name, password_hash, role)
            )

    def delete_user(username):
        with db() as conn:
            cur = conn.execute("DELETE FROM users WHERE username=?", (username,))
            return cur.rowcount > 0

    def list_users():
        with db() as conn:
            rows = conn.execute(
                "SELECT username,name,role,created_at FROM users ORDER BY created_at"
            ).fetchall()
            return [dict(r) for r in rows]

    def count_users():
        with db() as conn:
            return conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    def save_chat(username, symptom, result, severity=""):
        with db() as conn:
            conn.execute(
                "INSERT INTO chat_history (username,symptom,result,severity) VALUES (?,?,?,?)",
                (username or "guest", symptom[:500], result[:200], severity)
            )

    def get_chat_history(username, limit=20):
        with db() as conn:
            rows = conn.execute(
                "SELECT * FROM chat_history WHERE username=? ORDER BY created_at DESC LIMIT ?",
                (username, limit)
            ).fetchall()
            return [dict(r) for r in rows]

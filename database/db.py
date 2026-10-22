import sqlite3, json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "edupredict.db"

def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    with connect() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS users(
          id INTEGER PRIMARY KEY AUTOINCREMENT, full_name TEXT NOT NULL,
          email TEXT NOT NULL UNIQUE, roll_number TEXT UNIQUE,
          username TEXT NOT NULL UNIQUE, password_hash TEXT NOT NULL,
          role TEXT NOT NULL DEFAULT 'student' CHECK(role IN ('admin','student')),
          created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS predictions(
          id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL,
          student_name TEXT NOT NULL, roll_number TEXT, course TEXT NOT NULL,
          features_json TEXT NOT NULL, predicted_grade REAL NOT NULL,
          category TEXT NOT NULL, recommendations_json TEXT NOT NULL,
          created_at TEXT NOT NULL,
          FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_predictions_user ON predictions(user_id);
        CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions(created_at);
        """)

def get_user_by_username(username):
    with connect() as con:
        row = con.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        return dict(row) if row else None

def get_user_by_id(user_id):
    with connect() as con:
        row = con.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None

def insert_user(full_name, email, roll_number, username, password_hash, role="student"):
    with connect() as con:
        cur = con.execute("""INSERT INTO users(full_name,email,roll_number,username,password_hash,role,created_at)
            VALUES(?,?,?,?,?,?,?)""", (full_name,email,roll_number or None,username,password_hash,role,datetime.now(timezone.utc).isoformat()))
        return cur.lastrowid

def update_password(user_id, password_hash):
    with connect() as con:
        con.execute("UPDATE users SET password_hash=? WHERE id=?", (password_hash,user_id))

def save_prediction(user_id, student_name, roll_number, course, features, predicted_grade, category, recommendations):
    with connect() as con:
        cur = con.execute("""INSERT INTO predictions(user_id,student_name,roll_number,course,features_json,predicted_grade,category,recommendations_json,created_at)
            VALUES(?,?,?,?,?,?,?,?,?)""", (user_id,student_name,roll_number,course,json.dumps(features),float(predicted_grade),category,json.dumps(recommendations),datetime.now(timezone.utc).isoformat()))
        return cur.lastrowid

def list_predictions(user_id=None, limit=500):
    with connect() as con:
        if user_id is None:
            rows = con.execute("""SELECT p.*, u.username FROM predictions p JOIN users u ON u.id=p.user_id ORDER BY p.id DESC LIMIT ?""",(int(limit),)).fetchall()
        else:
            rows = con.execute("""SELECT p.*, u.username FROM predictions p JOIN users u ON u.id=p.user_id WHERE p.user_id=? ORDER BY p.id DESC LIMIT ?""",(int(user_id),int(limit))).fetchall()
    out=[]
    for row in rows:
        d=dict(row)
        d["features"] = json.loads(d.pop("features_json"))
        d["recommendations"] = json.loads(d.pop("recommendations_json"))
        out.append(d)
    return out

def list_students():
    with connect() as con:
        rows = con.execute("""SELECT u.id,u.full_name,u.email,u.roll_number,u.username,u.role,u.created_at,
            COUNT(p.id) prediction_count, AVG(p.predicted_grade) avg_grade
            FROM users u LEFT JOIN predictions p ON p.user_id=u.id
            GROUP BY u.id ORDER BY u.id DESC""").fetchall()
    return [dict(r) for r in rows]

def dashboard_stats():
    with connect() as con:
        students = con.execute("SELECT COUNT(*) n FROM users WHERE role='student'").fetchone()["n"]
        predictions = con.execute("SELECT COUNT(*) n FROM predictions").fetchone()["n"]
        avg = con.execute("SELECT AVG(predicted_grade) n FROM predictions").fetchone()["n"]
        support = con.execute("SELECT COUNT(*) n FROM predictions WHERE category='Needs support'").fetchone()["n"]
    return {"students":students,"predictions":predictions,"avg_grade":avg,"support_count":support}

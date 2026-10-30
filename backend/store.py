"""SQLite storage for raw setlists and logged predictions."""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "setlists.db"
_lock = threading.Lock()


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def init() -> None:
    with _lock, _conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS artists (
                mbid TEXT PRIMARY KEY,
                name TEXT,
                disambiguation TEXT,
                last_synced TEXT
            );
            CREATE TABLE IF NOT EXISTS setlists (
                id TEXT PRIMARY KEY,
                artist_mbid TEXT,
                event_date TEXT,      -- ISO yyyy-mm-dd
                raw TEXT
            );
            CREATE INDEX IF NOT EXISTS ix_setlists_artist ON setlists(artist_mbid, event_date);
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                artist_mbid TEXT,
                artist_name TEXT,
                created_at TEXT,
                show_date TEXT,
                city TEXT,
                country TEXT,
                is_festival INTEGER,
                payload TEXT,          -- full prediction JSON
                result TEXT            -- scoring JSON once the real setlist is known
            );
            """
        )


def upsert_artist(mbid: str, name: str, disambiguation: str, last_synced: str | None = None) -> None:
    with _lock, _conn() as c:
        c.execute(
            """INSERT INTO artists(mbid, name, disambiguation, last_synced) VALUES (?,?,?,?)
               ON CONFLICT(mbid) DO UPDATE SET name=excluded.name,
                 disambiguation=excluded.disambiguation,
                 last_synced=COALESCE(excluded.last_synced, artists.last_synced)""",
            (mbid, name, disambiguation, last_synced),
        )


def get_artist(mbid: str) -> dict | None:
    with _conn() as c:
        r = c.execute("SELECT * FROM artists WHERE mbid=?", (mbid,)).fetchone()
        return dict(r) if r else None


def list_artists() -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            """SELECT a.*, COUNT(s.id) AS n_setlists FROM artists a
               LEFT JOIN setlists s ON s.artist_mbid = a.mbid
               WHERE a.last_synced IS NOT NULL
               GROUP BY a.mbid ORDER BY a.last_synced DESC"""
        ).fetchall()
        return [dict(r) for r in rows]


def save_setlists(artist_mbid: str, setlists: list[dict]) -> int:
    """Insert or replace setlists. Returns how many ids were new."""
    with _lock, _conn() as c:
        existing = {r[0] for r in c.execute("SELECT id FROM setlists WHERE artist_mbid=?", (artist_mbid,))}
        new = 0
        for s in setlists:
            d, m, y = s["eventDate"].split("-")
            if s["id"] not in existing:
                new += 1
            c.execute(
                "INSERT OR REPLACE INTO setlists(id, artist_mbid, event_date, raw) VALUES (?,?,?,?)",
                (s["id"], artist_mbid, f"{y}-{m}-{d}", json.dumps(s)),
            )
        return new


def setlist_counts() -> dict[str, int]:
    """Number of stored setlists for every artist that has any."""
    with _conn() as c:
        return {m: n for m, n in c.execute("SELECT artist_mbid, COUNT(*) FROM setlists GROUP BY artist_mbid")}


def known_setlist_ids(artist_mbid: str) -> set[str]:
    with _conn() as c:
        return {r[0] for r in c.execute("SELECT id FROM setlists WHERE artist_mbid=?", (artist_mbid,))}


def load_setlists(artist_mbid: str) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT raw FROM setlists WHERE artist_mbid=? ORDER BY event_date", (artist_mbid,)
        ).fetchall()
        return [json.loads(r[0]) for r in rows]


def add_prediction(row: dict) -> int:
    with _lock, _conn() as c:
        cur = c.execute(
            """INSERT INTO predictions(artist_mbid, artist_name, created_at, show_date, city, country,
                                       is_festival, payload, result)
               VALUES (?,?,?,?,?,?,?,?,NULL)""",
            (
                row["artist_mbid"], row["artist_name"], row["created_at"], row["show_date"],
                row["city"], row["country"], int(row["is_festival"]), json.dumps(row["payload"]),
            ),
        )
        return cur.lastrowid


def list_predictions() -> list[dict]:
    with _conn() as c:
        rows = c.execute("SELECT * FROM predictions ORDER BY show_date DESC, id DESC").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["payload"] = json.loads(d["payload"])
            d["result"] = json.loads(d["result"]) if d["result"] else None
            out.append(d)
        return out


def set_prediction_result(pred_id: int, result: dict) -> None:
    with _lock, _conn() as c:
        c.execute("UPDATE predictions SET result=? WHERE id=?", (json.dumps(result), pred_id))

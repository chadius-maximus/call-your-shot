"""SQLite store and scoring for call-your-shot. Standard library only, no network."""

from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

PLUGIN = "call-your-shot"
DB_PATH: Optional[Path] = None  # tests point this at a temp file
RECORD_WINDOW = 20  # settled shots used for the bias figures
MIN_SHOTS_FOR_RECORD = 3
STALE_OPEN_SECONDS = 12 * 3600

_lock = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS shots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL DEFAULT '',
    turn_id TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    task TEXT NOT NULL,
    called_at REAL NOT NULL,
    pred_calls INTEGER NOT NULL,
    pred_seconds REAL NOT NULL,
    calls_before INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'open',
    actual_calls INTEGER,
    actual_seconds REAL,
    score INTEGER,
    settled_at REAL,
    by_tool TEXT,
    notified INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS stats (key TEXT PRIMARY KEY, value INTEGER NOT NULL DEFAULT 0);
"""


def _db_path() -> Path:
    if DB_PATH is not None:
        return Path(DB_PATH)
    from plugins.plugin_storage import plugin_data_dir  # follows the active Hermes profile
    return plugin_data_dir(PLUGIN) / "shots.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_db_path()), timeout=5)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def closeness(predicted: float, actual: float, floor: float) -> float:
    """1.0 for a perfect call, falling toward 0 as the two diverge in either direction."""
    p, a = max(float(predicted), floor), max(float(actual), floor)
    return min(p, a) / max(p, a)


def score_shot(pred_calls: int, actual_calls: int, pred_seconds: float, actual_seconds: float) -> int:
    """0-100: the mean closeness of the tool-call estimate and the time estimate."""
    return round(50 * (closeness(pred_calls, actual_calls, 1) + closeness(pred_seconds, actual_seconds, 1.0)))


def open_shot(*, session_id: str, turn_id: str, model: str, task: str, pred_calls: int,
              pred_seconds: float, calls_before: int, now: Optional[float] = None) -> int:
    with _lock, _connect() as conn:
        cur = conn.execute(
            "INSERT INTO shots (session_id, turn_id, model, task, called_at, pred_calls, pred_seconds,"
            " calls_before) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, turn_id, model, task, now or time.time(), pred_calls, pred_seconds, calls_before),
        )
        return int(cur.lastrowid)


def settle_shot(shot_id: int, *, actual_calls: int, actual_seconds: float,
                by_tool: Optional[dict] = None, now: Optional[float] = None) -> Optional[dict]:
    """Close an open shot with the measured numbers. Returns the settled row, or None if not open."""
    with _lock, _connect() as conn:
        row = conn.execute("SELECT * FROM shots WHERE id = ? AND status = 'open'", (shot_id,)).fetchone()
        if row is None:
            return None
        score = score_shot(row["pred_calls"], actual_calls, row["pred_seconds"], actual_seconds)
        conn.execute(
            "UPDATE shots SET status = 'settled', actual_calls = ?, actual_seconds = ?, score = ?,"
            " settled_at = ?, by_tool = ? WHERE id = ?",
            (actual_calls, actual_seconds, score, now or time.time(), json.dumps(by_tool or {}), shot_id),
        )
        return dict(conn.execute("SELECT * FROM shots WHERE id = ?", (shot_id,)).fetchone())


def void_shot(shot_id: int) -> None:
    """An interrupted or failed turn is not scored."""
    with _lock, _connect() as conn:
        conn.execute("UPDATE shots SET status = 'void', settled_at = ? WHERE id = ? AND status = 'open'",
                     (time.time(), shot_id))


def take_unnotified(session_id: str) -> Optional[dict]:
    """The newest settled shot of this session the agent has not been told about yet."""
    with _lock, _connect() as conn:
        row = conn.execute(
            "SELECT * FROM shots WHERE session_id = ? AND status = 'settled' AND notified = 0"
            " ORDER BY id DESC LIMIT 1", (session_id,)).fetchone()
        if row is None:
            return None
        conn.execute("UPDATE shots SET notified = 1 WHERE session_id = ? AND status = 'settled'", (session_id,))
        return dict(row)


def bump(key: str) -> None:
    with _lock, _connect() as conn:
        conn.execute("INSERT INTO stats (key, value) VALUES (?, 1)"
                     " ON CONFLICT(key) DO UPDATE SET value = value + 1", (key,))


def _geo_mean(ratios: list) -> float:
    return math.exp(sum(math.log(r) for r in ratios) / len(ratios))


def record() -> dict:
    """The track record over the most recent settled shots."""
    with _lock, _connect() as conn:
        conn.execute("UPDATE shots SET status = 'void' WHERE status = 'open' AND called_at < ?",
                     (time.time() - STALE_OPEN_SECONDS,))
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM shots WHERE status = 'settled' ORDER BY id DESC LIMIT ?", (RECORD_WINDOW,))]
        total = conn.execute("SELECT COUNT(*) FROM shots WHERE status = 'settled'").fetchone()[0]
        uncalled = conn.execute("SELECT value FROM stats WHERE key = 'uncalled_turns'").fetchone()
    out: dict[str, Any] = {"settled": int(total), "window": len(rows),
                           "uncalled_turns": int(uncalled[0]) if uncalled else 0}
    if not rows:
        return out
    out["avg_score"] = round(sum(r["score"] for r in rows) / len(rows))
    out["calls_bias"] = round(_geo_mean([max(r["actual_calls"], 1) / max(r["pred_calls"], 1) for r in rows]), 2)
    out["time_bias"] = round(_geo_mean([max(r["actual_seconds"], 1.0) / max(r["pred_seconds"], 1.0) for r in rows]), 2)
    return out


def recent(limit: int = 10) -> list:
    with _lock, _connect() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM shots WHERE status != 'void' ORDER BY id DESC LIMIT ?", (max(1, int(limit)),))]


def reset() -> int:
    with _lock, _connect() as conn:
        n = conn.execute("SELECT COUNT(*) FROM shots").fetchone()[0]
        conn.execute("DELETE FROM shots")
        conn.execute("DELETE FROM stats")
        conn.execute("DELETE FROM sqlite_sequence WHERE name = 'shots'")
        return int(n)

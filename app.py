#!/usr/bin/env python3
"""Structure Mapping Experiment Pack v0.1 local server."""

from __future__ import annotations

import argparse
import json
import mimetypes
import sqlite3
import time
import uuid
from contextlib import contextmanager
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
DEFAULT_DB = ROOT / "data" / "structure_mapping.sqlite3"


class Storage:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def connection(self):
        connection = self.connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self.connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    participant_id TEXT NOT NULL,
                    started_at REAL NOT NULL,
                    completed_at REAL,
                    app_version TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS trials (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(id),
                    task_type TEXT NOT NULL,
                    trial_index INTEGER NOT NULL,
                    started_at REAL NOT NULL,
                    completed_at REAL NOT NULL,
                    duration_ms REAL NOT NULL,
                    confidence INTEGER NOT NULL,
                    stimulus_json TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trial_id TEXT NOT NULL REFERENCES trials(id),
                    sequence INTEGER NOT NULL,
                    elapsed_ms REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                """
            )

    def create_session(self, participant_id: str) -> dict:
        session_id = str(uuid.uuid4())
        started_at = time.time()
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO sessions (id, participant_id, started_at, app_version)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, participant_id, started_at, "0.1.0"),
            )
        return {"session_id": session_id, "started_at": started_at}

    def save_trial(self, payload: dict) -> None:
        required = {
            "id", "session_id", "task_type", "trial_index", "started_at",
            "completed_at", "duration_ms", "confidence", "stimulus",
            "response", "metrics", "events",
        }
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"missing fields: {', '.join(sorted(missing))}")

        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO trials (
                    id, session_id, task_type, trial_index, started_at,
                    completed_at, duration_ms, confidence, stimulus_json,
                    response_json, metrics_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["id"], payload["session_id"], payload["task_type"],
                    int(payload["trial_index"]), float(payload["started_at"]),
                    float(payload["completed_at"]), float(payload["duration_ms"]),
                    int(payload["confidence"]),
                    json.dumps(payload["stimulus"], ensure_ascii=False),
                    json.dumps(payload["response"], ensure_ascii=False),
                    json.dumps(payload["metrics"], ensure_ascii=False),
                ),
            )
            connection.executemany(
                """
                INSERT INTO events (
                    trial_id, sequence, elapsed_ms, event_type, payload_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        payload["id"], index, float(event["elapsed_ms"]),
                        event["event_type"],
                        json.dumps(event.get("payload", {}), ensure_ascii=False),
                    )
                    for index, event in enumerate(payload["events"])
                ],
            )

    def complete_session(self, session_id: str) -> None:
        with self.connection() as connection:
            cursor = connection.execute(
                "UPDATE sessions SET completed_at = ? WHERE id = ?",
                (time.time(), session_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("session not found")

    def export_session(self, session_id: str) -> dict | None:
        with self.connection() as connection:
            session = connection.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if session is None:
                return None
            trials = connection.execute(
                "SELECT * FROM trials WHERE session_id = ? ORDER BY trial_index",
                (session_id,),
            ).fetchall()

            exported_trials = []
            for trial in trials:
                events = connection.execute(
                    """
                    SELECT sequence, elapsed_ms, event_type, payload_json
                    FROM events WHERE trial_id = ? ORDER BY sequence
                    """,
                    (trial["id"],),
                ).fetchall()
                exported_trials.append(
                    {
                        "id": trial["id"],
                        "task_type": trial["task_type"],
                        "trial_index": trial["trial_index"],
                        "started_at": trial["started_at"],
                        "completed_at": trial["completed_at"],
                        "duration_ms": trial["duration_ms"],
                        "confidence": trial["confidence"],
                        "stimulus": json.loads(trial["stimulus_json"]),
                        "response": json.loads(trial["response_json"]),
                        "metrics": json.loads(trial["metrics_json"]),
                        "events": [
                            {
                                "sequence": event["sequence"],
                                "elapsed_ms": event["elapsed_ms"],
                                "event_type": event["event_type"],
                                "payload": json.loads(event["payload_json"]),
                            }
                            for event in events
                        ],
                    }
                )

        return {"session": dict(session), "trials": exported_trials}


class AppHandler(BaseHTTPRequestHandler):
    storage: Storage

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/export":
            session_id = parse_qs(parsed.query).get("session_id", [""])[0]
            data = self.storage.export_session(session_id)
            if data is None:
                self.send_json({"error": "session not found"}, HTTPStatus.NOT_FOUND)
                return
            self.send_json(data)
            return
        self.serve_static(parsed.path)

    def do_POST(self) -> None:
        try:
            payload = self.read_json()
            if self.path == "/api/sessions":
                participant_id = str(payload.get("participant_id", "")).strip()
                if not participant_id:
                    raise ValueError("participant_id is required")
                self.send_json(
                    self.storage.create_session(participant_id),
                    HTTPStatus.CREATED,
                )
                return
            if self.path == "/api/trials":
                self.storage.save_trial(payload)
                self.send_json({"ok": True}, HTTPStatus.CREATED)
                return
            if self.path == "/api/sessions/complete":
                self.storage.complete_session(str(payload.get("session_id", "")))
                self.send_json({"ok": True})
                return
            self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except sqlite3.IntegrityError as error:
            self.send_json({"error": str(error)}, HTTPStatus.CONFLICT)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 5_000_000:
            raise ValueError("invalid request size")
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def send_json(self, data: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def serve_static(self, request_path: str) -> None:
        relative = "index.html" if request_path == "/" else request_path.lstrip("/")
        candidate = (STATIC_DIR / relative).resolve()
        static_root = STATIC_DIR.resolve()
        if static_root not in candidate.parents and candidate != static_root:
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not candidate.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = candidate.read_bytes()
        content_type, _ = mimetypes.guess_type(candidate.name)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        print(f"[server] {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = parser.parse_args()

    storage = Storage(args.db)
    handler = type("ConfiguredHandler", (AppHandler,), {"storage": storage})
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print("Structure Mapping Experiment Pack v0.1")
    print(f"Open http://{args.host}:{args.port}")
    print(f"Data: {args.db.resolve()}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

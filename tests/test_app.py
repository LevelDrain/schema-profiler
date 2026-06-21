import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.request import Request, urlopen

from http.server import ThreadingHTTPServer

from app import AppHandler, Storage


class StorageTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.storage = Storage(Path(self.temp.name) / "test.sqlite3")

    def tearDown(self):
        self.temp.cleanup()

    def test_session_trial_and_export(self):
        session = self.storage.create_session("self")
        trial = {
            "id": "trial-1",
            "session_id": session["session_id"],
            "task_type": "relation_mapping",
            "trial_index": 0,
            "started_at": 1.0,
            "completed_at": 2.0,
            "duration_ms": 1000,
            "confidence": 4,
            "stimulus": {"nodes": 3},
            "response": {"mapping": {"a": "b"}},
            "metrics": {"score": 1.0},
            "events": [{
                "elapsed_ms": 20,
                "event_type": "mapping_set",
                "payload": {"source_id": "a", "target_id": "b"},
            }],
        }
        self.storage.save_trial(trial)
        self.storage.complete_session(session["session_id"])

        exported = self.storage.export_session(session["session_id"])
        self.assertEqual(exported["session"]["participant_id"], "self")
        self.assertEqual(exported["trials"][0]["metrics"]["score"], 1.0)
        self.assertEqual(
            exported["trials"][0]["events"][0]["event_type"], "mapping_set"
        )

    def test_missing_session_returns_none(self):
        self.assertIsNone(self.storage.export_session("missing"))


class HttpTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        storage = Storage(Path(self.temp.name) / "http.sqlite3")
        handler = type("TestHandler", (AppHandler,), {"storage": storage})
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def request_json(self, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        request = Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request) as response:
            return response.status, json.load(response)

    def test_static_page_and_api_round_trip(self):
        with urlopen(self.base_url + "/") as response:
            page = response.read().decode()
            self.assertEqual(response.status, 200)
            self.assertIn("構造写像実験パック", page)

        status, session = self.request_json(
            "/api/sessions", {"participant_id": "http-test"}
        )
        self.assertEqual(status, 201)

        trial = {
            "id": "http-trial",
            "session_id": session["session_id"],
            "task_type": "structural_choice",
            "trial_index": 0,
            "started_at": 1,
            "completed_at": 2,
            "duration_ms": 1000,
            "confidence": 3,
            "stimulus": {"base": "x"},
            "response": {"choice_id": "y"},
            "metrics": {"score": 0},
            "events": [],
        }
        status, saved = self.request_json("/api/trials", trial)
        self.assertEqual(status, 201)
        self.assertTrue(saved["ok"])

        _, exported = self.request_json(
            f"/api/export?session_id={session['session_id']}"
        )
        self.assertEqual(len(exported["trials"]), 1)


if __name__ == "__main__":
    unittest.main()

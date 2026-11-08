import importlib.util
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location(
    "receipt_module", Path(__file__).parents[1] / "receipts.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = m.ReceiptStore(Path(self.tmp.name))
        self.id = self.store.begin(
            "source", "workspace", "channel", ["user", "assistant"], human=True
        )
        self.store.bind(self.id, "remote-session")

    def row(self):
        with sqlite3.connect(self.store.path) as c:
            return c.execute("SELECT status,error_class FROM turns").fetchone()

    def message(self, part, **kw):
        return SimpleNamespace(
            **dict(
                dict(
                    id="remote-" + str(part),
                    workspace_id="workspace",
                    session_id="remote-session",
                    metadata={"hermes_receipt": {"id": self.id, "part": part}},
                ),
                **kw,
            )
        )

    def test_out_of_order_remote_echo_acknowledges_all_parts(self):
        self.store.acknowledge([self.id], [self.message(1), self.message(0)])
        self.assertEqual(self.row(), ("acknowledged", None))
        self.assertEqual(self.store.path.stat().st_mode & 0o777, 0o600)

    def test_partial_response_is_unknown(self):
        self.store.acknowledge([self.id], [self.message(0)])
        self.assertEqual(self.row()[0], "unknown")

    def test_wrong_remote_session_cannot_confirm(self):
        with self.assertRaises(ValueError):
            self.store.acknowledge([self.id], [self.message(0, session_id="other")])
        self.assertEqual(self.row()[0], "pending")

    def test_duplicate_parts_cannot_confirm(self):
        with self.assertRaises(ValueError):
            self.store.acknowledge([self.id], [self.message(0), self.message(0)])

    def test_ambiguous_error_is_durable_and_has_no_message(self):
        self.store.uncertain([self.id], "TimeoutError")
        self.assertEqual(self.row(), ("unknown", "TimeoutError"))

    def test_profiles_are_separate(self):
        self.assertNotEqual(
            self.store.path, m.ReceiptStore(Path(self.tmp.name) / "other").path
        )

    def test_symlink_store_refused(self):
        bad = Path(self.tmp.name) / "bad"
        target = Path(self.tmp.name) / "target"
        target.write_text("preserve")
        store = m.ReceiptStore(bad)
        store.path.parent.mkdir(parents=True)
        store.path.symlink_to(target)
        with self.assertRaises(OSError):
            store.begin("s", "w", "k", ["user"], human=True)
        self.assertEqual(target.read_text(), "preserve")


if __name__ == "__main__":
    unittest.main()

"""Tests for the training history (no Qt / ultralytics needed):  python -m unittest discover assets/tests"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from history import (  # noqa: E402
    STATUS_FINISHED,
    STATUS_INTERRUPTED,
    STATUS_RUNNING,
    RunHistory,
    read_results_csv,
)

# Older ultralytics versions pad the header names with spaces, newer ones do not: both must work.
CSV_PADDED = (
    "                  epoch,         train/box_loss,         train/cls_loss,         train/dfl_loss,"
    "       metrics/mAP50(B),           val/box_loss,           val/cls_loss,           val/dfl_loss\n"
    "                      1,                  0.9,                  6.5,                  1.0,"
    "                     0,                  0.7,                  5.8,                  0.8\n"
    "                      2,                  0.6,                  4.0,                  0.9,"
    "                   0.5,                  0.7,                  3.0,                  0.8\n"
)
CSV_PLAIN = CSV_PADDED.replace(" ", "")


def make_run(history: RunHistory, **kwargs):
    defaults = dict(model="yolo11n.pt", epochs=2, batch=8, classes=["toy"], train_images=10, val_images=2)
    defaults.update(kwargs)
    return history.create(**defaults)


class HistoryTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.history = RunHistory(Path(self._tmp.name) / "trainings")

    def tearDown(self):
        self._tmp.cleanup()

    def test_results_csv_both_header_styles(self):
        for text in (CSV_PADDED, CSV_PLAIN):
            path = Path(self._tmp.name) / "results.csv"
            path.write_text(text)
            points = read_results_csv(path)
            self.assertEqual(len(points), 2)
            self.assertAlmostEqual(points[0].train_loss, 8.4, 5)
            self.assertAlmostEqual(points[1].val_loss, 4.5, 5)
            self.assertEqual(points[1].map50, 0.5)

    def test_create_list_and_summary(self):
        record = make_run(self.history, name="my first try")
        self.assertEqual(record.status, STATUS_RUNNING)
        (record.path / "results.csv").write_text(CSV_PLAIN)
        (record.path / "weights").mkdir()
        (record.path / "weights" / "best.pt").write_bytes(b"x")
        (record.path / "weights" / "last.pt").write_bytes(b"x")
        (record.path / "events.out.tfevents.1.pc").write_bytes(b"x")
        self.history.finish(record, STATUS_FINISHED, 12.0, "log text")

        runs = self.history.list_runs()
        self.assertEqual([r.run_id for r in runs], [record.run_id])
        self.assertEqual(runs[0].status, STATUS_FINISHED)
        self.assertEqual(runs[0].summary()["best_map50"], 0.5)
        self.assertEqual(runs[0].summary()["epochs_done"], 2)
        self.assertEqual(runs[0].log_path.read_text(), "log text")
        self.assertTrue(runs[0].has_weights)
        self.assertEqual(len(self.history.usable_runs()), 1)
        self.assertFalse((record.path / "weights" / "last.pt").exists())  # tidied
        self.assertEqual(list(record.path.glob("events.out.tfevents.*")), [])

    def test_unique_ids_and_newest_first(self):
        first = make_run(self.history)
        second = make_run(self.history)
        self.assertNotEqual(first.run_id, second.run_id)
        self.assertEqual(self.history.list_runs()[0].run_id, max(first.run_id, second.run_id))

    def test_delete_removes_everything(self):
        record = make_run(self.history)
        (record.path / "results.csv").write_text(CSV_PLAIN)
        self.history.delete(record.run_id)
        self.assertFalse(record.path.exists())
        self.assertEqual(self.history.list_runs(), [])
        self.history.delete("does_not_exist")  # no error

    def test_running_runs_become_interrupted(self):
        record = make_run(self.history)
        self.history.mark_interrupted()
        self.assertEqual(self.history.get(record.run_id).status, STATUS_INTERRUPTED)

    def test_discard_and_folders_without_run_json_are_ignored(self):
        record = make_run(self.history)
        (self.history.root / "random_folder").mkdir()
        self.history.discard(record)
        self.assertEqual(self.history.list_runs(), [])


if __name__ == "__main__":
    unittest.main()

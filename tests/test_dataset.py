"""Tests for the dataset layer (no Qt / ultralytics needed):  python -m unittest discover tests"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import COCO_CLASS_NAMES  # noqa: E402
from dataset import Box, DatasetError, DatasetStore  # noqa: E402


class DatasetStoreTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = DatasetStore(Path(self._tmp.name) / "ds")

    def tearDown(self):
        self._tmp.cleanup()

    def _frame(self):
        return np.random.default_rng(0).integers(0, 255, (1200, 1600, 3), dtype=np.uint8)

    def test_coco_has_80_names(self):
        self.assertEqual(len(COCO_CLASS_NAMES), 80)

    def test_class_rules(self):
        self.store.add_class("my toy")
        self.assertEqual(self.store.classes(), ["my_toy"])
        with self.assertRaises(DatasetError):
            self.store.add_class("MY_TOY")
        with self.assertRaises(DatasetError):
            self.store.add_class("chair")
        with self.assertRaises(DatasetError):
            self.store.add_class("bad/name")
        with self.assertRaises(DatasetError):
            self.store.add_class("   ")

    def test_yolo_line_roundtrip_uses_id_80_plus(self):
        box = Box(1, 0.1, 0.2, 0.5, 0.6)
        self.assertTrue(box.to_yolo_line().startswith("81 "))
        back = Box.from_yolo_line(box.to_yolo_line(), num_custom=2)
        self.assertAlmostEqual(back.x1, 0.1, 5)
        self.assertAlmostEqual(back.y2, 0.6, 5)
        self.assertIsNone(Box.from_yolo_line("3 0.5 0.5 0.1 0.1", num_custom=2))  # COCO id

    def test_annotation_lifecycle(self):
        self.store.add_class("toy")
        item = self.store.add_frame(self._frame())
        self.assertEqual(item.location, "inbox")
        self.assertLessEqual(max(__import__("cv2").imread(str(item.path)).shape[:2]), 960)

        saved = self.store.save_boxes(item, [Box(0, 0.2, 0.2, 0.6, 0.7)])
        self.assertIn(saved.location, ("train", "val"))
        self.assertTrue(saved.path.exists() and not item.path.exists())
        self.assertEqual(len(self.store.load_boxes(saved)), 1)

        stats = self.store.stats()
        self.assertEqual(stats.instances, (1,))
        self.assertEqual(stats.train_images + stats.val_images, 1)

        back = self.store.save_boxes(saved, [])  # removing every box un-annotates the image
        self.assertEqual(back.location, "inbox")
        self.assertFalse(self.store.label_path(saved).exists())

        self.store.delete_item(back)
        self.assertEqual(self.store.list_items(), [])

    def test_data_yaml_has_80_plus_n_names(self):
        self.store.add_class("toy")
        with self.assertRaises(DatasetError):
            self.store.write_data_yaml()  # no annotated images yet
        item = self.store.add_frame(self._frame())
        self.store.save_boxes(item, [Box(0, 0.1, 0.1, 0.4, 0.4)])
        data = yaml.safe_load(self.store.write_data_yaml().read_text())
        self.assertEqual(len(data["names"]), 81)
        self.assertEqual(data["names"][0], "person")
        self.assertEqual(data["names"][80], "toy")
        self.assertTrue(Path(data["path"]).is_absolute())


if __name__ == "__main__":
    unittest.main()

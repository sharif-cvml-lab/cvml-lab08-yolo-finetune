"""Generate the small starter dataset shipped with the lab (instructor tool).

The starter set contains three synthetic "toys" that are NOT COCO classes:
    toy_star (yellow star), toy_cube (blue cube), toy_ring (purple ring).
Images are drawn procedurally, so every bounding box is exact. They are only a
format example - students are expected to add their own real photos and boxes.

Usage (from the repository root):
    python tools/make_starter_dataset.py            # adds starter_*.jpg files
    python tools/make_starter_dataset.py --clean    # first delete ALL dataset images/labels
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DATASET_DIR, NUM_COCO_CLASSES  # noqa: E402
from dataset import DatasetStore, write_jpeg  # noqa: E402

CLASSES = ["toy_star", "toy_cube", "toy_ring"]
WIDTH, HEIGHT = 640, 480
NUM_IMAGES = 36
VAL_INDICES = {0, 5, 10, 15, 20, 25, 30, 35}  # 8 validation images


def make_background(rng: np.random.Generator) -> np.ndarray:
    top = rng.integers(60, 200, size=3).astype(np.float32)
    bottom = rng.integers(60, 200, size=3).astype(np.float32)
    ramp = np.linspace(0, 1, HEIGHT, dtype=np.float32)[:, None, None]
    image = top * (1 - ramp) + bottom * ramp
    image = np.repeat(image, WIDTH, axis=1)
    image += rng.normal(0, 6, size=image.shape)
    image = np.clip(image, 0, 255).astype(np.uint8)
    for _ in range(int(rng.integers(3, 8))):  # grey clutter so boxes are not trivial
        shade = int(rng.integers(90, 170))
        x, y = int(rng.integers(0, WIDTH)), int(rng.integers(0, HEIGHT))
        if rng.random() < 0.5:
            w, h = int(rng.integers(30, 160)), int(rng.integers(10, 80))
            cv2.rectangle(image, (x, y), (x + w, y + h), (shade,) * 3, -1)
        else:
            x2, y2 = int(rng.integers(0, WIDTH)), int(rng.integers(0, HEIGHT))
            cv2.line(image, (x, y), (x2, y2), (shade,) * 3, int(rng.integers(2, 6)))
    return cv2.GaussianBlur(image, (5, 5), 0)


def render_star(size: int) -> tuple[np.ndarray, np.ndarray]:
    canvas = int(size * 1.5)
    patch = np.zeros((canvas, canvas, 3), np.uint8)
    mask = np.zeros((canvas, canvas), np.uint8)
    centre, outer, inner = canvas / 2, size / 2, size / 2 * 0.45
    points = []
    for k in range(10):
        radius = outer if k % 2 == 0 else inner
        angle = -math.pi / 2 + k * math.pi / 5
        points.append((centre + radius * math.cos(angle), centre + radius * math.sin(angle)))
    polygon = np.array(points, np.int32)
    cv2.fillPoly(patch, [polygon], (40, 200, 255))
    cv2.fillPoly(mask, [polygon], 255)
    cv2.polylines(patch, [polygon], True, (10, 120, 200), max(2, size // 25))
    cv2.circle(patch, (int(centre), int(centre)), max(2, size // 9), (140, 235, 255), -1)
    return patch, mask


def render_cube(size: int) -> tuple[np.ndarray, np.ndarray]:
    canvas = int(size * 1.5)
    patch = np.zeros((canvas, canvas, 3), np.uint8)
    mask = np.zeros((canvas, canvas), np.uint8)
    cx, cy, r = canvas / 2, canvas / 2, size / 2
    hexagon = [(cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a)))
               for a in (-90, -30, 30, 90, 150, 210)]
    centre = (cx, cy)
    faces = [
        ([hexagon[0], hexagon[1], centre, hexagon[5]], (255, 170, 90)),   # top
        ([hexagon[5], centre, hexagon[3], hexagon[4]], (200, 110, 30)),   # left
        ([hexagon[1], hexagon[2], hexagon[3], centre], (140, 70, 10)),    # right
    ]
    for pts, color in faces:
        polygon = np.array(pts, np.int32)
        cv2.fillPoly(patch, [polygon], color)
        cv2.fillPoly(mask, [polygon], 255)
        cv2.polylines(patch, [polygon], True, (60, 30, 5), max(1, size // 40))
    return patch, mask


def render_ring(size: int) -> tuple[np.ndarray, np.ndarray]:
    canvas = int(size * 1.5)
    patch = np.zeros((canvas, canvas, 3), np.uint8)
    mask = np.zeros((canvas, canvas), np.uint8)
    centre = (canvas // 2, canvas // 2)
    outer, inner = size // 2, int(size * 0.2)
    cv2.circle(patch, centre, outer, (170, 60, 150), -1)
    cv2.circle(mask, centre, outer, 255, -1)
    cv2.circle(patch, centre, inner, (0, 0, 0), -1)
    cv2.circle(mask, centre, inner, 0, -1)
    cv2.ellipse(patch, centre, (outer - 4, outer - 4), 0, 200, 290, (230, 150, 220), max(2, size // 20))
    cv2.circle(patch, centre, outer, (100, 30, 90), max(2, size // 30), cv2.LINE_AA)
    return patch, mask


RENDERERS = [render_star, render_cube, render_ring]


def place_object(background, class_id, rng, taken):
    """Draw one object; return its pixel box (x1, y1, x2, y2) or None if no free spot."""
    size = int(rng.integers(90, 190))
    patch, mask = RENDERERS[class_id](size)
    angle = float(rng.uniform(-35, 35)) if class_id != 2 else float(rng.uniform(0, 360))
    centre = (patch.shape[1] / 2, patch.shape[0] / 2)
    matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)
    patch = cv2.warpAffine(patch, matrix, (patch.shape[1], patch.shape[0]))
    mask = cv2.warpAffine(mask, matrix, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    bx1, by1, bx2, by2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    bw, bh = bx2 - bx1, by2 - by1
    if bw >= WIDTH - 10 or bh >= HEIGHT - 10:
        return None
    for _ in range(40):
        x = int(rng.integers(5, WIDTH - bw - 5))
        y = int(rng.integers(5, HEIGHT - bh - 5))
        box = (x, y, x + bw, y + bh)
        if all(box[2] < t[0] or box[0] > t[2] or box[3] < t[1] or box[1] > t[3] for t in taken):
            crop_patch = patch[by1:by2, bx1:bx2]
            crop_mask = mask[by1:by2, bx1:bx2] > 0
            region = background[y:y + bh, x:x + bw]
            region[crop_mask] = crop_patch[crop_mask]
            return box
    return None


def clean(root: Path) -> None:
    for sub in ("images/train", "images/val", "labels/train", "labels/val", "inbox"):
        for path in (root / sub).glob("*"):
            if path.is_file() and path.name != ".gitkeep":
                path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--clean", action="store_true", help="delete all existing dataset images/labels first")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    store = DatasetStore(DATASET_DIR)
    if args.clean:
        clean(store.root)
    store.classes_file.write_text("\n".join(CLASSES) + "\n", encoding="utf-8")

    rng = np.random.default_rng(args.seed)
    counts = [0] * len(CLASSES)
    for index in range(NUM_IMAGES):
        image = make_background(rng)
        primary = index % len(CLASSES)
        wanted = [primary]
        if rng.random() < 0.35:
            wanted.append(int((primary + 1 + rng.integers(0, len(CLASSES) - 1)) % len(CLASSES)))
        taken: list[tuple[int, int, int, int]] = []
        lines = []
        for class_id in wanted:
            box = place_object(image, class_id, rng, taken)
            if box is None:
                continue
            taken.append(box)
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2 / WIDTH, (y1 + y2) / 2 / HEIGHT
            w, h = (x2 - x1) / WIDTH, (y2 - y1) / HEIGHT
            lines.append(f"{NUM_COCO_CLASSES + class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            counts[class_id] += 1
        if not lines:
            continue
        split = "val" if index in VAL_INDICES else "train"
        name = f"starter_{index:03d}"
        write_jpeg(store.images_dir / split / f"{name}.jpg", image)
        (store.labels_dir / split / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    stats = store.stats()
    print(f"train images: {stats.train_images}, val images: {stats.val_images}")
    for name, count in zip(CLASSES, counts):
        print(f"  {name}: {count} boxes")


if __name__ == "__main__":
    main()

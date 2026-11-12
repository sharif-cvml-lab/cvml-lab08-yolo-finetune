from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

APP_NAME = "CVML Lab 08 - YOLO Fine-Tuning"
ORGANIZATION_NAME = "Sharif CVML Lab"

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parent

# Everything that is not Python code lives in assets/:
#   assets/dataset/   your images + boxes (YOLO format)
#   assets/weights/   pretrained YOLO weights (downloaded here automatically)
#   assets/outputs/   everything the program CREATES (one folder per training)
ASSETS_DIR = ROOT_DIR / "assets"
DATASET_DIR = ASSETS_DIR / "dataset"
WEIGHTS_DIR = ASSETS_DIR / "weights"
OUTPUTS_DIR = ASSETS_DIR / "outputs"
TRAININGS_DIR = OUTPUTS_DIR / "trainings"
RUN_NAME = "train"  # default folder name; the GUI replaces it with a unique name per training

# --------------------------------------------------------------------------- #
# Camera / UI
# --------------------------------------------------------------------------- #
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
TARGET_FPS = 30

DEFAULT_CAMERA_INDEX = 0
DEFAULT_THEME = "dark"
DEFAULT_SETTINGS_MESSAGE = "Settings: Ready"

# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
NUM_COCO_CLASSES = 80          # custom classes are numbered 80, 81, 82, ...
MAX_IMAGE_SIDE = 960           # imported / captured images are downscaled to this
VAL_EVERY = 5                  # roughly 1 image out of 5 goes to the validation split

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")

IMAGE_FILTER = (
    "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff);;"
    "All Files (*)"
)
FILE_FILTER = (
    "Media Files (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff "
    "*.mp4 *.avi *.mov *.mkv *.webm *.m4v);;"
    "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff);;"
    "Videos (*.mp4 *.avi *.mov *.mkv *.webm *.m4v);;"
    "All Files (*)"
)

# --------------------------------------------------------------------------- #
# Training (kept small on purpose: the goal is to *see* fine-tuning work on a CPU)
# --------------------------------------------------------------------------- #
WEIGHT_CHOICES = ("yolov8n.pt", "yolo11n.pt")
DEFAULT_WEIGHTS = "yolov8n.pt"

IMG_SIZE = 320                 # training image size (small = fast on CPU)
OPTIMIZER = "AdamW"
LEARNING_RATE = 0.005          # only the new class outputs are trained, so a larger rate is fine

DEFAULT_EPOCHS = 200
DEFAULT_BATCH_SIZE = 8

# --------------------------------------------------------------------------- #
# Inference (Before / After)
# --------------------------------------------------------------------------- #
INFER_IMG_SIZE = 416
DEFAULT_CONFIDENCE = 30        # percent

COCO_CLASS_NAMES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)


@dataclass(frozen=True)
class TrainingParameters:
    weights: str
    epochs: int
    batch_size: int

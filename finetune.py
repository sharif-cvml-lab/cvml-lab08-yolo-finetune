"""Lab 08 - Fine-tune a YOLO detector on your own dataset.

=====================================================================
  THIS IS THE ONLY FILE YOU HAVE TO EDIT.
  Everything else (GUI, annotation tool, dataset, threads, inference)
  is already implemented.
=====================================================================

What the GUI does for you
-------------------------
1. Step 1 builds the dataset (images + YOLO labels, classes numbered 80, 81, ...).
2. When you press "Train Model" in Step 2, the GUI writes `dataset/data.yaml` and calls
   `train_custom_detector(...)` below in a background thread.
3. Step 3 compares the original YOLO (Base) with the model you return (Custom).

What you must write
-------------------
Complete the five TODOs inside `train_custom_detector`. Until you do, the application
still runs: Step 1 and Step 3 (Base model only) work, and pressing "Train Model" shows
a message telling you which TODO is missing.

Expected pipeline
-----------------
    load pretrained YOLO -> keep the COCO head -> train ONLY the new class outputs
                         -> model.train(...) -> return the path of the best weights (best.pt)

Why the classes start at 80, and why we freeze almost everything
----------------------------------------------------------------
The pretrained weights know the 80 COCO classes (ids 0-79). Your classes are ids 80, 81, ...
Fine-tuning normally makes the network "forget" old classes. Here we avoid that: the
backbone, the neck, the box regression and the 80 COCO outputs stay frozen, and only the
output rows of YOUR classes are trained. The model can then still detect every COCO object
exactly as before *and* your new objects.
"""
from __future__ import annotations

from pathlib import Path

from config import IMG_SIZE, LEARNING_RATE, OPTIMIZER, RUN_NAME
from model_utils import freeze_all_but_new_classes, get_device, preserve_coco_head
from training_reporter import TrainingReporter

# Given: safe settings for a small CPU-friendly run inside a background thread.
#   workers=0        -> no dataloader worker processes (they break inside a GUI thread on Windows)
#   plots=False      -> skip saving many diagnostic figures (faster)
#   verbose=False    -> the GUI shows its own progress
#   warmup_epochs=0  -> our runs are short; do not spend them warming up
#   close_mosaic=5   -> disable mosaic augmentation for the last 5 epochs
SAFE_TRAIN_KWARGS = dict(
    workers=0,
    plots=False,
    verbose=False,
    warmup_epochs=0,
    close_mosaic=5,
    exist_ok=True,
    seed=0,
)


def train_custom_detector(
    dataset_config: str,
    pretrained_weights: str,
    epochs: int,
    batch_size: int,
    output_dir: str,
    reporter: TrainingReporter | None = None,
) -> str:
    """Fine-tune a pretrained YOLO model on the custom dataset.

    Args:
        dataset_config:     path to `dataset/data.yaml` (80 COCO names + your classes).
        pretrained_weights: path/name of the starting weights, e.g. "yolov8n.pt".
        epochs:             number of passes over the training set.
        batch_size:         images per optimisation step.
        output_dir:         folder where ultralytics stores the run (`project=`).
        reporter:           given by the GUI; lets it display progress (already wired below).

    Returns:
        The path (str) of the best weights file, i.e. `<run folder>/weights/best.pt`.
    """
    from ultralytics import YOLO  # imported here so the rest of the app starts quickly

    # ------------------------------------------------------------------ #
    # TODO 1 - Load the pretrained detector.
    #   Create a `YOLO` object from `pretrained_weights`.
    #   Expected: `model` is a ultralytics.YOLO instance (COCO, 80 classes).
    # ------------------------------------------------------------------ #
    model = None
    if model is None:
        raise NotImplementedError("TODO 1: load the pretrained model with YOLO(pretrained_weights).")

    # Given: lets the GUI show epoch / loss / progress. Keep this line.
    if reporter is not None:
        reporter.attach(model)

    # ------------------------------------------------------------------ #
    # TODO 2 - Transfer learning: copy what the model already knows.
    #   Your dataset has 80 + N classes, so the classification head has a new shape and
    #   ultralytics would re-initialise it. `preserve_coco_head(model)` (model_utils.py)
    #   returns a callback that copies the pretrained head into the new one.
    #   Register it on the event "on_pretrain_routine_end" with
    #   `model.add_callback(event, callback)`.
    # ------------------------------------------------------------------ #
    # (write your code here)
    if not getattr(model, "coco_head_preserved", False):
        raise NotImplementedError("TODO 2: register preserve_coco_head(model) with model.add_callback(...).")

    # ------------------------------------------------------------------ #
    # TODO 3 - Do not forget: train ONLY the new classes.
    #   Call `freeze_all_but_new_classes(model)` (model_utils.py). It freezes the backbone,
    #   neck, box branch and the 80 COCO outputs, so only the output rows of your classes
    #   are updated. Think about it: why does this make forgetting impossible, and what
    #   is the price? (read its docstring)
    # ------------------------------------------------------------------ #
    # (write your code here)
    if not getattr(model, "new_classes_only", False):
        raise NotImplementedError("TODO 3: call freeze_all_but_new_classes(model).")

    # ------------------------------------------------------------------ #
    # TODO 4 - Fine-tune.
    #   Call `model.train(...)` with these arguments (use the names you receive plus the
    #   constants imported from config.py) and also pass **SAFE_TRAIN_KWARGS:
    #       data=...      the dataset yaml                 epochs=...   number of epochs
    #       batch=...     the batch size                   imgsz=...    IMG_SIZE
    #       device=...    get_device()                     project=...  output_dir
    #       name=...      RUN_NAME                         optimizer=... OPTIMIZER
    #       lr0=...       LEARNING_RATE
    # ------------------------------------------------------------------ #
    # (write your code here)
    if getattr(model, "trainer", None) is None:
        raise NotImplementedError("TODO 4: call model.train(...) with the arguments listed above.")

    # ------------------------------------------------------------------ #
    # TODO 5 - Return the best weights.
    #   After training, `model.trainer.save_dir` is the run folder. The best checkpoint is
    #   `<save_dir>/weights/best.pt`. Return its path as a string.
    # ------------------------------------------------------------------ #
    best_weights = None
    if best_weights is None:
        raise NotImplementedError("TODO 5: return str(Path(model.trainer.save_dir) / 'weights' / 'best.pt').")
    return str(best_weights)

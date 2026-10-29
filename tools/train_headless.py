"""Run finetune.train_custom_detector without the GUI (handy for debugging / grading).

    python tools/train_headless.py --epochs 5 --batch 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_WEIGHTS, RUNS_DIR  # noqa: E402
from dataset import DatasetStore  # noqa: E402
from model_utils import resolve_weights  # noqa: E402
from training_reporter import TrainingReporter  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    import finetune

    yaml_path = DatasetStore().write_data_yaml()
    reporter = TrainingReporter(lambda event: print(event) if event["type"] == "epoch" else None)
    best = finetune.train_custom_detector(
        dataset_config=str(yaml_path),
        pretrained_weights=resolve_weights(args.weights),
        epochs=args.epochs,
        batch_size=args.batch,
        output_dir=str(RUNS_DIR),
        reporter=reporter,
    )
    print("Best weights:", best)


if __name__ == "__main__":
    main()

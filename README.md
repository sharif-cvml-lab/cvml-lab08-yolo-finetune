# CVML Lab 08 - YOLO Fine-Tuning on a Custom Dataset

In the previous session you used a YOLO model pretrained on COCO. In this lab you **teach it new objects**:
you collect photos, draw the bounding boxes yourself, fine-tune the model, and compare
**Base YOLO vs Custom YOLO** side by side.

Almost the whole application is already implemented. **You only edit one file: `finetune.py`.**

## Theory covered

Dataset, annotation, train / validation split, transfer learning, pretrained model, fine-tuning,
epoch, batch size, loss, evaluation (mAP).

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

The pretrained weights should be placed in `weights/`. 

Note: Some new version of Ultralytics automatically donwlaod the latest nano version of YOLO.

The pretrained weights should be placed in weights/.
Download link: https://my.sharif.edu/s/6qLsEdPobLPeSwp

Note: Some new versions of Ultralytics automatically download the latest nano version of YOLO.

## The three steps in the application

| Step | What you do |
| --- | --- |
| **1 Dataset Builder** | Add photos (or capture them with the webcam), pick a class, **drag with the mouse to draw boxes**, move quickly between images, save. Images and YOLO labels are written into `dataset/` for you. |
| **2 Train** | Choose epochs / batch size and press **Train Model**. Training runs in a background thread; you see progress, epoch, training loss, validation loss, mAP50, elapsed time and a loss curve. |
| **3 Before / After** | Two panels: **Base YOLO** and **Custom YOLO** on the same webcam / image / video. |

The project runs *before* you write any code: Step 1 and Step 3 (Base YOLO) work, and pressing **Train Model**
shows which TODO in `finetune.py` is still missing.

## Your task: `finetune.py`

Implement the body of

```python
train_custom_detector(dataset_config, pretrained_weights, epochs, batch_size, output_dir, reporter=None)
```

Pipeline: **load pretrained YOLO -> copy the COCO head -> train only the new class outputs -> `model.train(...)` -> return `best.pt`**.
There are five `TODO` blocks with hints. `finetune.py` is re-imported on every run, so you can edit it and press
**Train Model** again without restarting the application.

### How forgetting is avoided

Normal fine-tuning updates the whole network and the model slowly forgets old classes. Here only the
output rows of **your** classes are trainable: backbone, neck, box regression, BatchNorm statistics and the 80 COCO
outputs are frozen (`freeze_all_but_new_classes` in `model_utils.py`). The 80 COCO scores are therefore exactly those of
the pretrained model. The trade-off is that new objects must be recognised from frozen COCO features: distinctive
objects work well, objects that look very similar to each other are harder. If a new class is weak, add more varied
photos, then raise the epochs or `LEARNING_RATE` in `config.py`.

### Class numbering

COCO has 80 classes (ids 0-79). **Your classes are numbered 80, 81, 82, ...**, so the fine-tuned model can still
detect COCO objects and your new ones. `dataset/data.yaml` (generated automatically before each run) lists
the 80 COCO names followed by your names.

## The starter dataset and your own objects

`dataset/` already contains a tiny starter set (36 images) of three synthetic toys - `toy_star`, `toy_cube`,
`toy_ring` - so you can see the format and train immediately:

```
dataset/
  classes.txt            custom class names (one per line)
  images/train, val/     JPEG images
  labels/train, val/     one .txt per image:  <class id> <cx> <cy> <w> <h>   (normalised, ids 80+)
  inbox/                 images waiting for boxes (not used for training)
  data.yaml              generated before training
```

Now make it yours:

1. In Step 1 type a name for your object (e.g. `my_toy`, `lab_chair`) and press **Add**. Choose something that is
   **not** a COCO class (the tool refuses exact COCO names such as `chair`).
2. Photograph the object from many angles, distances and backgrounds (webcam button or **Add Images...**).
   Around 15-30 photos per object is enough for this lab.
3. Draw a box around every instance, move on (A/D), and save.
4. Open the dataset folder: your images and labels are there next to the starter ones.
5. Train, then check Step 3.

Images are automatically downscaled to at most 960 px; roughly one image in five goes to the validation split.

## Keyboard shortcuts

| Key | Action |
| --- | --- |
| `Ctrl+1` / `Ctrl+2` / `Ctrl+3` | Switch step |
| `A` / `D` (or arrows) | Previous / next image |
| `1`-`9` | Choose class (relabels the selected box) |
| `Ctrl+S` | Save annotations (moving to another image also saves) |
| `Ctrl+Z` | Undo last box |
| `Delete` | Delete selected box (right-click also deletes a box) |
| `Space` | Capture photo (Step 1, webcam on) / pause-resume (Step 3) |
| `T` | Toggle dark / light theme |
| `R` | Reset parameters to defaults (the dataset is never touched) |
| `Q` | Quit |

## Repository structure

```
main.py                 entry point
config.py               constants (paths, training defaults, COCO names)
finetune.py             <-- THE ONLY FILE YOU EDIT
gui.py                  main window, theme, shortcuts, settings
gui_dataset.py          Step 1 page            annotation_canvas.py  box editor widget
gui_train.py            Step 2 page            train_thread.py       background training thread
gui_compare.py          Step 3 page            inference_thread.py   Base vs Custom inference
dataset.py              dataset store (YOLO format, data.yaml)
model_utils.py          device, weights, COCO head transfer, train-only-new-classes freezing
training_reporter.py    progress callbacks for the GUI
video_thread.py         camera helpers         styles.py             dark / light themes
tools/                  make_starter_dataset.py, train_headless.py
tests/                  dataset tests (python -m unittest discover tests)
```

Settings (theme, camera index, epochs, batch size, confidence) are stored with `QSettings` and restored on start-up.

## Questions to think about

- Why do we start from pretrained weights instead of random weights?
- Why does freezing everything except the new class outputs make forgetting impossible? What is the price (hint: the new objects must be recognised from frozen COCO features)?
- Compare Base and Custom in Step 3 on a COCO object (e.g. a person or a cup). Are the detections the same as
  Base? Why? Then temporarily skip TODO 3 (remove its check) and train again: what changes?
- What happens with 3 photos per class? With photos taken only against one background?
- How do training loss and validation loss behave when you train for many more epochs?

## Troubleshooting

| Problem | Fix |
| --- | --- |
| "ultralytics is not installed" | `pip install -r requirements.txt` |
| Weights cannot be downloaded | Copy `yolov8n.pt` into `weights/` |
| Webcam does not open | Change *Camera index* in the sidebar; close other apps using the camera |
| Training is slow | Lower epochs / batch size; use fewer, smaller images |
| Start over with the starter data | `python tools/make_starter_dataset.py --clean` (deletes **all** dataset images and labels) |

## For instructors

- `tools/make_starter_dataset.py` regenerates the synthetic starter set (exact boxes). Replace it with real photos
  of a toy you bring to class if you prefer.
- `tools/train_headless.py` runs `finetune.py` without the GUI (useful for checking submissions).
- The reference solution is distributed separately (`finetune_solution.py`); copy it over `finetune.py`.

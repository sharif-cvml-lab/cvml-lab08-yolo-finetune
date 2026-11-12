# Lab08_YOLOFineTuning

Welcome to the eighth assignment of the Computer Vision and Machine Learning lab course!

This project provides a PyQt6 application for fine-tuning a YOLO object detector on your own objects. In the previous session you used a YOLO model pretrained on COCO. In this lab you **teach it new objects**: you collect photos, draw the bounding boxes yourself, fine-tune the model, and compare **Base YOLO vs Custom YOLO** side by side. The goal of this lab is to understand datasets, annotation, transfer learning, fine-tuning, loss curves, and detection accuracy (mAP) through an interactive application.

## 🎯 Assignment Objectives

In this lab, you will complete one specific function located in `finetune.py`:

```
train_custom_detector(dataset_config, pretrained_weights, epochs, batch_size, output_dir, reporter=None)
```

Students should modify only `finetune.py`. Do not modify `main.py`, `config.py`, the `gui_*.py` files, or any other project file.

The expected pipeline is:

```
Pretrained YOLO (80 COCO classes)
     ↓
Copy the COCO head into the bigger head (80 + N outputs)
     ↓
Freeze everything except the N new class outputs
     ↓
model.train(...) on your dataset
     ↓
Return best.pt
```

There are five `TODO` blocks with hints. `finetune.py` is re-imported on every run, so you can edit it and press **Train Model** again without restarting the application.

### Theory covered

Dataset, annotation, train / validation split, transfer learning, pretrained model, fine-tuning, epoch, batch size, loss, overfitting, evaluation (mAP).

### Your tasks

1. **Explore the starter dataset.** `assets/dataset/` already contains 36 images of three synthetic toys (`toy_star`, `toy_cube`, `toy_ring`) with their boxes. Open Step 1, flip through the images and look carefully at how the boxes are placed around the objects. Then train once (Step 2) and check the result in Step 3.
2. **Build your own dataset.** Find a suitable object in the lab, photograph it from many angles, import the photos, define one or more new classes, draw the boxes, train, and examine the result.
3. **Experiment.** Train with different settings (number of epochs, `yolov8n` vs `yolo11n`, more or fewer photos) and compare the trainings in the **History** tab.

### Tips for a good dataset

- **Variety matters most.** Change the angle, the distance and the surroundings as much as you can. 15-30 varied photos are better than 100 almost identical ones.
- Let **other objects appear in the background**, both objects that are among the 80 COCO classes and objects that are not.
- While labelling, draw a box **only around your own new object** with your new class. Do **not** label objects that belong to older classes (person, cup, ...) even if they are visible in the photo.
- Draw a box around **every** instance of your object in the image, and keep the boxes tight.

## ✨ Features

- Three-step workflow: **Dataset Builder**, **Train**, **Before / After**.
- Annotation canvas: drag to draw boxes, click to select, right-click to delete, relabel a box by choosing another class.
- Photos from files or from the webcam, with a **Capture** button and a space-bar shortcut.
- Add and **delete classes** at any time (the boxes of a deleted class are removed with it).
- Tick many images and **delete them in one step**.
- Training in a background thread: progress bar, epoch, losses, mAP50, elapsed time and ETA.
- Loss curves with axis titles and a legend (training vs validation), plus a short explanation of how to read them.
- **Training history:** every training is saved (model, settings, curves, log). Compare trainings, reopen them in Step 3, or delete them.
- Before / After: Base YOLO and Custom YOLO on the same webcam, image or video, and a list to pick **any** of your trainings.
- Runs without a webcam (offline images and videos).
- Dark mode and light mode, settings restored automatically, Reset button, FPS indicator.

## 🛠️ Setup & Installation

1. Clone the repository and navigate to the project directory.

2. Create a virtual environment (recommended):

```
python -m venv venv
```

Activate it on Linux/macOS:

```
source venv/bin/activate
```

On Windows PowerShell:

```
venv\Scripts\Activate.ps1
```

3. Install the required dependencies:

```
pip install -r requirements.txt
```

4. Pretrained weights: the application downloads the nano models (`yolov8n.pt`, `yolo11n.pt`) into `assets/weights/` automatically the first time they are needed (internet required once).

5. Run the application:

```
python main.py
```

## 🎮 Using the Application

### Step 1 - Dataset Builder

- **Add Images...** or **Webcam + Capture (Space)** put photos into the list. Photos wait in the list as `○ not annotated` until you draw at least one box on them.
- Type a name in the class box and press **Add**. Use a name that is **not** a COCO class (the tool refuses names such as `chair`).
- **Drag** to draw a box, move to the next image with `D` (or `A` for the previous one). Moving to another image saves the boxes automatically; one image in five goes to the validation split.
- **Delete Selected Class** removes the selected class and every box that uses it. Images that lose their last box return to the `○ not annotated` state. The application tells you how many boxes are affected before it asks for confirmation.
- **Delete Image** removes the current image. To delete **many** images, tick their check boxes (or use **Tick All**) and press **Delete Ticked**: you confirm only once.
- While a training is running, Step 1 is read-only (the trainer is reading the dataset files).

### Step 2 - Train

The **Training** tab shows the training that is running:

- Choose the pretrained model, the number of epochs and the batch size. The optional **Run name** helps you recognise the training later.
- **Loss** measures how wrong the predictions are, so lower is better. The **green** curve is the *training loss* (on the images the model learns from) and the **orange** curve is the *validation loss* (on images it never trains on). The vertical axis is the sum of the box, class and dfl losses. If green keeps falling while orange rises, the model is memorising your photos (overfitting).
- **mAP50** is the detection accuracy on the validation images (0 to 1, higher is better).

The **History** tab lists every training you ever ran:

- Select one row to see its settings and log, or several rows (`Ctrl` / `Shift`) to draw their curves together. Choose what to compare: validation loss, training loss or mAP50.
- **Open in Before / After** loads that training's model in Step 3.
- **Delete Selected** removes the trainings completely (results, log and saved model), so the History always matches what is on disk.

### Step 3 - Before / After

Two panels show **Base YOLO** and **Custom YOLO** on the same webcam / image / video. Pick any of your trainings in the **Custom model** list; the Base panel automatically uses the pretrained model that training started from. Try your new object in front of the camera: the Base model cannot name it, the Custom model can. Then compare a COCO object (for example a person or a cup): both models should agree.

### Class numbering

COCO has 80 classes (ids 0-79). **Your classes are numbered 80, 81, 82, ...**, so the fine-tuned model can still detect COCO objects and your new ones. `assets/dataset/data.yaml` (generated automatically before each training) lists the 80 COCO names followed by your names. When you delete a class, the classes after it move up by one id.

### How forgetting is avoided

Normal fine-tuning updates the whole network and the model slowly forgets old classes. Here only the output rows of **your** classes are trainable: backbone, neck, box regression, BatchNorm statistics and the 80 COCO outputs are frozen (`freeze_all_but_new_classes` in `model_utils.py`). The 80 COCO scores are therefore exactly those of the pretrained model. The trade-off is that new objects must be recognised from frozen COCO features: distinctive objects work well, objects that look very similar are harder. If a new class is weak, add more varied photos, then raise the epochs or `LEARNING_RATE` in `config.py`.

## 📂 Repository Structure

```
cvml-lab08-yolo-finetune/
├── .gitignore
├── README.md
├── requirements.txt
├── main.py
├── config.py
├── gui.py
├── gui_dataset.py
├── gui_train.py
├── gui_compare.py
├── gui_common.py
├── gui_plot.py
├── annotation_canvas.py
├── dataset.py
├── history.py
├── model_utils.py
├── training_reporter.py
├── train_thread.py
├── inference_thread.py
├── video_thread.py
├── thread_utils.py
├── styles.py
├── finetune.py             # The only file students should modify
└── assets/
    ├── dataset/            # Your images and boxes (YOLO format)
    │   ├── classes.txt     #   custom class names, one per line
    │   ├── images/         #   train/ and val/ JPEG images
    │   ├── labels/         #   one .txt per image: <class id> <cx> <cy> <w> <h>
    │   └── inbox/          #   images waiting for boxes
    ├── weights/            # Pretrained YOLO weights (downloaded here)
    ├── outputs/            # Everything the program CREATES
    │   └── trainings/      #   one folder per training (see below)
    ├── tools/              # Helper scripts for instructors
    └── tests/              # Unit tests
```

### Where do the results go?

Everything the program creates is stored in `assets/outputs/trainings/`, **one folder per training**, named `<date>_<time>_<run name or model_epochs>`:

```
assets/outputs/trainings/20261112_141846_yolo11n_e50/
├── run.json          # model, epochs, batch size, classes, image counts, status
├── log.txt           # the log shown in the Train page
├── results.csv       # losses and mAP for every epoch
├── args.yaml         # every ultralytics argument that was used
└── weights/best.pt   # the fine-tuned model (used in Step 3)
```

Deleting a training in the History tab deletes this folder. You can also delete folders by hand: the History tab only lists folders that contain a `run.json`.

### Module responsibilities

- `main.py` - application entry point only.
- `config.py` - constants (paths, training defaults, COCO class names).
- `finetune.py` - the student implementation of the fine-tuning pipeline.
- `gui.py` - main window, theme, shortcuts, settings.
- `gui_dataset.py` - Step 1 page; `annotation_canvas.py` - the box editor widget.
- `gui_train.py` - Step 2 page (live training and history); `gui_plot.py` - curve plot widget.
- `gui_compare.py` - Step 3 page.
- `dataset.py` - dataset store (YOLO format, classes, `data.yaml`).
- `history.py` - the training history (one folder per training).
- `model_utils.py` - device, weights, COCO head transfer, train-only-new-classes freezing.
- `train_thread.py`, `training_reporter.py` - background training and progress callbacks.
- `inference_thread.py` - Base vs Custom inference; `video_thread.py` - camera helpers.
- `thread_utils.py` - safe stopping of background threads.
- `styles.py` - dark / light themes.

Settings (theme, camera index, epochs, batch size, confidence, selected model) are stored with `QSettings` and restored on start-up.

## ⌨️ Keyboard Shortcuts

- `Ctrl+1` / `Ctrl+2` / `Ctrl+3` - Switch step
- `A` / `D` (or arrows) - Previous / next image
- `1`-`9` - Choose class (relabels the selected box)
- `Ctrl+S` - Save annotations (moving to another image also saves)
- `Ctrl+Z` - Undo last box
- `Delete` - Delete selected box (right-click also deletes a box)
- `Space` - Capture photo (Step 1, webcam on) / Pause - Resume (Step 3)
- `T` - Toggle dark / light theme
- `R` - Reset parameters to defaults (the dataset and the history are never touched)
- `Q` - Quit

## 🤔 Questions to Think About

- Why do we start from pretrained weights instead of random weights?
- Why does freezing everything except the new class outputs make forgetting impossible? What is the price?
- Compare Base and Custom in Step 3 on a COCO object (a person or a cup). Are the detections the same? Why? Then temporarily skip TODO 3 (remove its check) and train again: what changes?
- What happens with 3 photos per class? With photos taken only against one background?
- How do the training loss and the validation loss behave when you train for many more epochs? Use the History tab to compare a short and a long training.
- Does `yolo11n` reach a better mAP50 than `yolov8n` on your dataset?

## Troubleshooting

- **"ultralytics is not installed"** - run `pip install -r requirements.txt`.
- **Weights cannot be downloaded** - copy `yolov8n.pt` / `yolo11n.pt` into `assets/weights/`.
- **Webcam does not open** - change *Camera index* in the sidebar and close other apps that use the camera. The application also works with image and video files.
- **Training is slow** - lower the epochs / batch size, or use fewer images.
- **Start over with the starter data** - `python assets/tools/make_starter_dataset.py --clean` (deletes **all** dataset images and labels).

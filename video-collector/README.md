# Uzbek Sign Language (UzSL) Video Collector  
  
**Python:** `3.9.23`  
**Author:** Custom-built for isolated dynamic sign collection  
**Purpose:** Record **videos** + **per-frame MediaPipe landmarks** in a structured, scalable way  

---

## Overview

This is a **CLI-based 3-stage data collection system** for **Uzbek Sign Language (UzSL)** using **MediaPipe Holistic**.  

It supports:  
- Multiple **signers** (`signer01`, `signer02`, ...)  
- Dynamic **sign word list** (add words anytime)  
- **Multiple repetitions** per sign  
- **Real-time feedback** (countdown, rep count, tree view)  
- **Automatic folder & file management**  


## 3 Stages of Workflow

| Stage | Description | Key Press |
|------|-------------|-----------|
| **1. Signer Selection** | Choose or create `signerXX` | Type ID |
| **2. Sign Word Selection** | Pick from numbered list (green = recorded) | Number / `a` / `b` |
| **3. Recording** | Press `s` → countdown → record **32 frames** | `s` = again, `d` = done |

> **After 32 frames**, only then can you press `s` or `d`.

---

## Video Collector Structure (Files)

<pre>
video-collector/
│
├── mod01_config.py          # Global settings (paths, FPS, landmarks)
├── mod02_storage.py         # Folder creation, progress, tree view
├── mod03_recorder.py        # OpenCV + MediaPipe (main processing)
├── mod04_ui.py              # CLI menus (signer, sign, post-recording)
├── mod05_main.py            # Entry point – run this!
│
├──environment-video-collector.yml  # for reproducing video-collector environment
│
└──dataset-checks/           # unit testing of dataset
    ├── 01_check_sign_count_per_signer.py   # Verifies that each signer directory contains the expected number of sign folders
    ├── 02_count_repetitions_per_sign.py    # Counts the total number of repetitions (rep-* folders) for each sign across all signers
    ├── 03_check_rep_consistency.py         # Data consistency checker for each repetitions present in landmarks and videos folder
    ├── 04_visualize_landmarks.py           # Loads one frame-XX.npy file and displays its 3D landmarks in an interactive Plotly plot
    ├── 05_verify_npy_shapes.py             # Verifies that every frame-*.npy file in the dataset has the exact shape (1662,)
    └── 06_trash_unwanted_sign.py           # Safely moves selected sign folders to the macOS Trash for all signers
</pre>


## Dataset Folder Structure
<pre>
video-collector/
│
└── Data_Numpy_Arrays_RSL_UzSL/
    └── signer{XX}/                      # e.g., signer01, signer02, ..., signer10
        └── {sign}/                      # e.g., assalomu_alaykum, bahor, ...
            ├── landmarks/
            │   └── rep-{XX}/            # e.g., rep-0, rep-1, ..., rep-XX (repetitions)
            │       ├── frame-00.npy
            │       ├── frame-01.npy
            │       ├── ...
            │       └── frame-XX.npy
            └── videos/
                └── rep-{XX}/            # e.g., rep-0, ..., rep-XX
                    └── video.mp4
</pre>

> Each `.npy` has 1662 float32 values face (468×3) + pose (33×4) + hands (2×21×3)

---

## Install Dependencies and Activate Environment
For the sake of separation of concerns, the `video-collector` stage of the project has its own dedicated environment. In order to able to do video collection with required dependencies, run the following commands to create the environment:
```shell
conda env create -f video-collector/environment-video-collector.yml

conda activate video_collector_env
```

## Run video collector
Run the following command inside of the `video-collector` folder:
```shell
python mod05_main.py
```

---

## User Interface
<pre>
=== CURRENT DATASET TREE ===
  signer01  [2/5 signs recorded]
     [✓] nima  (reps: 3)
     [✓] salom  (reps: 1)
     [ ] rahmat (reps: 0)
     ...

=== SIGNER MENU ===
Enter a new signer ID (e.g. signer03) or pick an existing one.
Signer ID: signer01

=== SIGN WORD LIST ===
 1. nima
 2. salom
 3. rahmat
 4. yaxshi
[a] Add new word  [b] Back

Select: 1

Press **s** in the camera window to start repetition 1 of **nima**

[Camera shows: "Press 's' to start"]

→ Press **s** → 5-second countdown → records 32 frames

Finished repetition 1 for sign **nima**
[s] Record another    [d] Done → back to list
</pre>

---

## Dataset validation
The [`video-collector/dataset-checks/`](./dataset-checks/) directory contains helper scripts to validate and maintain the integrity of the processed dataset located in `video-collector/Data_Numpy_Arrays_RSL_UzSL` folder (after pruning to the final 50 cleaned signs).


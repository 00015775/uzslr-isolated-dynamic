# Uzbek Sign Language (UzSL) Video Collector  
  
**Python:** `3.9.23`  
**Author:** Custom-built for isolated dynamic sign collection  
**Purpose:** Record **videos** + **per-frame MediaPipe landmarks** in a structured, scalable way  

> **Note:** The `video-collector` stage is **the first step** of this project if you are collecting Uzbek Sign Language (UzSL) videos from real-world signers. This step records videos and extracts **per-frame MediaPipe landmarks**, storing them in a structured format under `video-collector/Data_Numpy_Arrays_RSL_UzSL`. It ensures that all signers, signs, and repetitions are properly captured and validated.

> Once the dataset is collected and verified in `video-collector/Data_Numpy_Arrays_RSL_UzSL`, the **next step** is [`dataset-prep`](../dataset-prep/). The `dataset-prep` stage reorganizes these raw landmarks, copies them into a root `/data/` folder, and splits them into **train, validation, and test sets**, preparing the dataset for subsequent preprocessing and model training.

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
.
└──video-collector/
    │
    ├── mod01_config.py          # Global settings (paths, FPS, landmarks)
    ├── mod02_storage.py         # Folder creation, progress, tree view
    ├── mod03_recorder.py        # OpenCV + MediaPipe (main processing)
    ├── mod04_ui.py              # CLI menus (signer, sign, post-recording)
    ├── mod05_main.py            # Main script that runs the entire video collection workflow
    │
    ├──environment-video-collector.yml          # for reproducing video-collector environment
    │
    └──dataset-checks/                          # unit testing of video-collector/Data_Numpy_Arrays_RSL_UzSL
        ├── 01_check_sign_count_per_signer.py   # Verifies that each signer directory contains the expected number of sign folders
        ├── 02_count_repetitions_per_sign.py    # Counts the total number of repetitions (rep-* folders) for each sign across all signers
        ├── 03_check_rep_consistency.py         # Data consistency checker for each repetitions present in landmarks and videos folder
        ├── 04_visualize_landmarks.py           # Loads one frame-XX.npy file and displays its 3D landmarks in an interactive Plotly plot
        ├── 05_verify_npy_shapes.py             # Verifies that every frame-*.npy file in the dataset has the exact shape (1662,)
        └── 06_trash_unwanted_sign.py           # Safely moves selected sign folders to the macOS Trash for all signers
</pre>


## Dataset Folder Structure
<pre>
.
└──video-collector/
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

Each script serves a specific purpose:

- [**01_check_sign_count_per_signer.py**](./dataset-checks/01_check_sign_count_per_signer.py) → Checks that every signer folder contains the expected number of sign subfolders. Ensures no signs are missing per signer.
- [**02_count_repetitions_per_sign.py**](./dataset-checks/02_count_repetitions_per_sign.py) → Counts all repetitions (`rep-*` folders) for each sign across all signers. Detects missing or extra repetitions.
- [**03_check_rep_consistency.py**](./dataset-checks/03_check_rep_consistency.py) → Verifies that each repetition folder contains the same number of frames in both `landmarks/` and `videos/`. Ensures consistency between video and landmark data.
- [**04_visualize_landmarks.py**](./dataset-checks/04_visualize_landmarks.py) → Loads a single `frame-XX.npy` file and renders its 3D landmarks using Plotly for quick visual inspection.
- [**05_verify_npy_shapes.py**](./dataset-checks/05_verify_npy_shapes.py) → Confirms that every `.npy` file has the expected shape `(1662,)`. Detects corrupted or incorrect landmark files.
- [**06_trash_unwanted_sign.py**](./dataset-checks/06_trash_unwanted_sign.py) → Safely moves selected sign folders to the macOS Trash, allowing removal of unwanted signs without accidental data loss.


## Customizing for Other Sign Languages

> **Note:** Although this `video-collector` is designed for collecting Uzbek Sign Language (UzSL) videos, it can be adapted to collect videos for **any other sign language**.

To customize:

1. **Change signs:**  
   - Open [`video-collector/mod01_config.py`](./mod01_config.py).  
   - Replace the existing `DEFAULT_SIGNS` dictionary (50 Uzbek signs) with your desired signs.  
   - **Important:** Keep the dictionary name as `DEFAULT_SIGNS`, since it is referenced in other `video-collector` scripts.  
   - **Additional note:** If you change `DEFAULT_SIGNS`, you must also update the corresponding `DEFAULT_SIGNS` used in all `dataset-checks` scripts for both `video-collector` (`dataset-checks/`) and `dataset-prep` ([`dataset-prep/dataset-checks/`](../dataset-prep/dataset-checks/)) so that checks remain consistent.
   - Also update `DEFAULT_SIGNS` in other `.py` files that use it.
   - **Tip:** To avoid manually searching through files, use GitHub search within this repository to locate all occurrences of `DEFAULT_SIGNS` variable in `.py` files that may need updating.


2. **Adjust recording parameters as needed:**  
   - `VIDEO_DEVICE` → select the camera device (default is the MacOS main webcam).  
   - `FPS` → frames per second (default: 30).  
   - `FRAMES_PER_REP` → number of frames per repetition (default: 32).  
   - `FRAME_WIDTH`, `FRAME_HEIGHT` → resolution of the recording (default: 1280, 720).
   - `COUNTDOWN_SECONDS` → delay before recording starts (default: 2).

3. **Optionally, change the dataset folder name by modifying:**  
```python
   DATA_ROOT = "./Data_Numpy_Arrays_RSL_UzSL"
```
If you rename the dataset folder, you must also update all scripts that reference it, including:
- [`step01_reorganize_dataset.py`](../dataset-prep/step01_reorganize_dataset.py) (for dataset reorganization)
- Any `dataset-checks` scripts in [`video-collector/dataset-checks/`](./dataset-checks/)
- Possibly other `.py` files in `video-collector` and `dataset-prep` where the folder path is used.

> **Note:** Failing to update all references will cause the scripts to fail or produce incorrect outputs. Following these steps allows you to collect videos for any set of signs while keeping the landmark extraction, dataset validation, and subsequent dataset preparation ([`dataset-prep`](../dataset-prep/)) steps fully compatible.
>
> **Tip:** To avoid manually searching through files, use GitHub search within this repository to locate all occurrences of `DEFAULT_SIGNS` and `Data_Numpy_Arrays_RSL_UzSL` in `.py` files that may need updating or changing.


## Next Steps: Dataset Preparation

> **Note:** Once the dataset is collected and verified, the [**dataset-prep**](../dataset-prep/) phase must be followed.  
> The `dataset-prep` step copies and reorganizes the extracted landmarks from `video-collector/Data_Numpy_Arrays_RSL_UzSL` into a `/data/` folder located at the root level and splits them into **train, validation, and test sets**. It prepares the dataset for further data preprocessing and model training.  
> 
> **Environment Recommendation:** Although `dataset-prep` mainly uses Python's built-in libraries, it is advised to run all scripts in the **same environment** used for video-collection phase ([`video_collector_env`](./environment-video-collector.yml)) to ensure compatibility and avoid any potential library conflicts.


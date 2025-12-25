# Uzbek Sign Language Recognition (UzSLR)

This repository aims to develop a **machine learning model for recognizing isolated dynamic Uzbek Sign Language (UzSL)** from video data.  

To achieve this, the repository provides a **full pipeline** that includes:

1. **Video Collection** ([`video-collector`](./video-collector/))  
2. **Dataset Preparation** ([`dataset-prep`](./dataset-prep/))  
3. **Data Preprocessing** (_upcoming_)  
4. **Model Training and Evaluation** (_future work_)
5. **Publication** (_future work: docs on `GitHub wiki` and paper in `LaTeX`_)

> Note: `video-collector` and `dataset-prep` are foundational steps — they generate a clean, structured dataset that the model will later use.

---

## Conda Environments

To ensure reproducibility, we use a dedicated environment for `video-collector` and `dataset-prep`:

```bash
# Change directory to where "environment-video-collector.yml" is located
cd ./video-collector

# Create and activate the environment
conda env create -f environment-video-collector.yml
conda activate video_collector_env
```
> [!TIP]
> This environment should be active for all scripts in `video-collector` and `dataset-prep`.
>
> The remaining steps may use their own dedicated conda environments to isolate dependencies, which will be provided in their respective folders once implemented.

---

## Phase 1: Video Collection ([`video-collector`](./video-collector/))

**Purpose:** Record signers performing signs and extract per-frame MediaPipe landmarks.

### Key Features:
- Supports multiple signers and sign words.
- Multiple repetitions per sign.
- Real-time feedback (countdown, rep count, CLI tree view).
- Automatic folder and file management.

### Output Dataset Structure
<pre>
.
└─ video-collector/Data_Numpy_Arrays_RSL_UzSL/
    └─ signerXX/
      └─ sign_name/
          ├─ landmarks/rep-XX/frame-XX.npy
          └─ videos/rep-XX/video.mp4
</pre>

### Run Video Collection

```shell
cd ./video-collector
python mod05_main.py
```
> After recording, run dataset checks in [`video-collector/dataset-checks/`](./video-collector/dataset-checks/) to ensure integrity.

> [!TIP]
> For the full understanding of how to run or modify `video-collector` and `video-collector/dataset-checks/`, it is **strongly advised to read** [`video-collector/README.md`](./video-collector/README.md)

---

## Phase 2: Dataset Preparation ([`dataset-prep`](./dataset-prep/))

**Purpose:** Reorganize raw landmarks for model training and split dataset into train/validation/test sets.

### Key Steps:
- Copy landmarks into `/data/` (pre-split dataset).
- Validate frame counts and repetitions.
- Split dataset into train (80%), validation (10%), and test (10%) sets.
- Verify dataset splits.

### Post-Splitting Dataset Structure
<pre>
.
└─ data/
    ├─ train/{sign_name}/rep-{XX}/frame-{XX}.npy
    ├─ validation/{sign_name}/rep-{XX}/frame-{XX}.npy
    └─ test/{sign_name}/rep-{XX}/frame-{XX}.npy
</pre>

### Run Dataset Preparation

> [!CAUTION]
> Please read **[`dataset-prep/README.md`](./dataset-prep/README.md) carefully before running these scripts.**  
> These scripts modify **GB-sized datasets** and may accidentally overwrite or delete your dataset.

```shell
# Assuming that you are at the root directory of the repository

# Move to the dataset preparation directory
cd ./dataset-prep
python step01_reorganize_dataset.py

# Go to dataset integrity check scripts (before splitting)
cd ./dataset-checks
python 01_check_frames.py
python 02_count_repetitions.py

# Return to dataset-prep to perform train/val/test split
cd ..
python step02_train_val_test_split.py

# Re-enter dataset-checks for post-split validation
cd ./dataset-checks
python 03_verify_dataset_splits.py
python 04_check_frames_after_dataset_splits.py
```

---

## Phase 3: Data Preprocessing (_upcoming_)

This phase will handle normalization, augmentation, and feature engineering of the landmark data to prepare it for model training. Detailed instructions and scripts will be provided once this phase is finalized.


---


## Phase 4: Model Training and Evaluation (_future work_)

This phase will focus on training and evaluating machine learning models for **isolated dynamic Uzbek Sign Language recognition** using the prepared and preprocessed landmark dataset.


---


## Phase 5: Publication (_future work: docs on `GitHub wiki` and paper in `LaTeX`_)

This phase will focus on documenting and analyzing the results of the project. The aim is to contribute to the broader sign language recognition and low-resource language research communities.


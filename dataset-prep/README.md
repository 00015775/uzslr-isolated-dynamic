# Dataset Folder Reorganization

> **Note:** This README is intentionally detailed and lengthy. The information provided here is essential for correctly preparing the dataset, without following these steps and specifications, the model will not have access to a valid or usable dataset.

The functions in the `/dataset-prep/` folder are intended to be executed **after** completing all functions and steps in the [`/video-collector/`](../video-collector/) folder. Before running any commands here, ensure that the video collection and landmark extraction steps in `/video-collector/` are fully completed.

## Relationship with `video-collector`

The `dataset-prep` folder reorganizes an **existing dataset**, assuming the dataset is located at: `video-collector/Data_Numpy_Arrays_RSL_UzSL`


This folder structure is generated during the `video-collector` phase. While organized, it is **not optimal for direct model training**, because:

- Input features are individual `frame-XX.npy` files, not videos at all.
- Video-related folders are ignored in subsequent steps.
- The dataset requires splitting into `train/validation/test` sets for proper model training.

### Original Dataset Structure (from `video-collector`)
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


## Dataset Reorganization

The main purpose of `dataset-prep` is to **copy only the `frame-XX.npy` files** from the `landmarks/` folder into a new `/data/` folder at the root of the project. This step prepares a **pre-splitting dataset**, which will then be divided into training, validation, and test sets.

### Pre-Splitting Dataset Structure
<pre>
.          
└── data/
    └── {sign}/                      # e.g., assalomu_alaykum, bahor, birga, ...
         └── rep-{XX}/               # e.g., rep-0, rep-1, ..., rep-XX (repetitions)
             ├── frame-00.npy
             ├── frame-01.npy
             ├── ... 
             └── frame-XX.npy
</pre>

This structure is created using [`step01_reorganize_dataset.py`](../dataset-prep/step01_reorganize_dataset.py).

## Train/Validation/Test Split

After reorganizing the dataset, [`step02_train_val_test_split.py`](../dataset-prep/step02_train_val_test_split.py) is used to split the data:

- **Training:** 80%
- **Validation:** 10%
- **Test:** 10%
- **Stratified random split:** By sign (all signs appear in every split)
- **Unit of Split:** Entire `rep-XX` folders

### Post-Split Dataset Structure
<pre>
.
└── data/
    ├── train/
    │   └── {sign}/                  # e.g., assalomu_alaykum, bahor, ...
    │        └── rep-{XX}/           # e.g., rep-0, rep-01, ..., rep-XX (repetitions)
    │            ├── frame-00.npy
    │            ├── ...
    │            └── frame-XX.npy
    │
    ├── validation/         # used for hyperparameter tunning
    │   └── {sign}/
    │        └── rep-{XX}/        
    │            ├── frame-00.npy
    │            ├── ...
    │            └── frame-XX.npy
    │
    └── test/               # used for the final model results
        └── {sign}/
             └── rep-{XX}/       
                 ├── frame-00.npy
                 ├── ...
                 └── frame-XX.npy
</pre>

**Reference:** Information learned from [HuggingFace - File names and splits](https://huggingface.co/docs/hub/datasets-file-names-and-splits)

> **Note:** `step02_train_val_test_split.py` copies landmarks into the new splits without removing the pre-split data. The dataset size will **double** temporarily. After verifying the splits, you may remove the original `{sign}` folders in `/data/`.
>
> **Environment:** Use the [`video_collector_env`](../video-collector/environment-video-collector.yml) environment for both `video-collector` and `dataset-prep` steps. Although only basic Python libraries are required for `dataset-prep`, using the same environment ensures compatibility.
>
> **Caution:** These scripts operate on GB-sized datasets and are intended to run **only once**.

## Order of Execution

> **Important:** Before running any Python function, ensure that you **change the working directory** to the folder where the intended `.py` file is located.  
>
> The file paths inside each script are **relative**, so running a script from a different directory may cause file or folder paths not to be found. In some cases, the script may still run without raising an error, but the outputs can be incorrect (such as, reporting missing data or empty results even though the data exists).  
>
> **Always execute the scripts from their own directory to ensure correct behavior!**


#### Example (Unix-like systems)

```shell
# Change directory to where the ".py" function is located
cd /path/to/where/function/resides

# Run the Python script from its own directory
python intended_function.py
```

> The following commands assume that you are **already inside the directory of the corresponding `.py` file before executing it**.

### Phase 1: (`video-collector`)

1. Complete the `video-collector` steps defined in [`video-collector/README.md`](../video-collector/README.md) and ensure `Data_Numpy_Arrays_RSL_UzSL` is properly created.

2. Run unit tests in [`video-collector/dataset-checks`](../video-collector/dataset-checks) to ensure no missing data.

```shell
python <unit_test_script_name>.py
```

### Phase 2: (`dataset-prep`)

3. Run [`dataset-prep/step01_reorganize_dataset.py`](./step01_reorganize_dataset.py) to copy landmarks to `/data/`.
```shell
python step01_reorganize_dataset.py
```

4. Run the following dataset integrity checks defined in the [`dataset-prep/dataset-checks/`](./dataset-checks/) folder, in the provided order:
   - [`01_check_frames.py`](./dataset-checks/01_check_frames.py) → ensures 32 frames per repetition.
   - [`02_count_repetitions.py`](./dataset-checks/02_count_repetitions.py) → verifies total repetitions per sign.
```shell
python 01_check_frames.py

# after verifying outputs, then run this:
python 02_count_repetitions.py
```
5. Run [`dataset-prep/step02_train_val_test_split.py`](./step02_train_val_test_split.py) to split the dataset.
```shell
python step02_train_val_test_split.py
```
6. Verify split integrity with:
   - [`03_verify_dataset_splits.py`](./dataset-checks/03_verify_dataset_splits.py) → checks that all signs exist in every split.
   - [`04_check_frames_after_dataset_splits.py`](./dataset-checks/04_check_frames_after_dataset_splits.py) → confirms 32 frames per repetition.

```shell
python 03_verify_dataset_splits.py

# after verifying outputs, then run this:
python 04_check_frames_after_dataset_splits.py
```

7. Optionally remove pre-split `{sign}` subfolders to save space.

## Git Ignore Information

Due to their large size, the following folders are included in `.gitignore`:
<pre>
video-collector/Data_Numpy_Arrays_RSL_UzSL/     # ~3 GB
data/                                           # ~1.1 GB pre-split, ~2.2 GB post-split
</pre>

## Dataset Info

- **Total signs:** 50  
- **Repetitions per sign:** 38 signs × 52 reps, 12 signs × 51 reps  
- **File format:** NumPy arrays (`.npy`)  
- **Glossary:** "rep" = repetition  
- **Total repetitions:** 2,588 
- **Total frames:** 82,816

### Post-Splitting Dataset Stats

- **Training repetitions:** 2,038 (38 signs x 41 reps, 12 signs x 40 reps)  
- **Validation repetitions:** 250 (5 reps per sign)  
- **Test repetitions:** 300 (6 reps per sign)  

## Dataset Limitations

- **Handedness Imbalance:** The dataset contains **10 signers** in total. Most signers are right-handed (8/10), while only `signer06` and `signer07` are left-handed. Additionally, some signs produced by `signer05` are partially left-handed. This imbalance may introduce a bias toward right-handed signs in models trained on this dataset.

- **No signer-independence:** The dataset is **not split by signer**, so repetitions from the same signer can appear in training, validation, and test sets. This may cause the model to learn signer-specific patterns rather than general sign features.


> **Final Note:** No data preprocessing is applied at this stage.  
> This stage only organizes, checks, and splits the raw landmark data so it is correctly structured and reliable.  
> Although no preprocessing is applied here, this step is **foundational** — all subsequent data preprocessing and model training stages depend on the correctness of the dataset produced at this stage of `dataset-prep`.


---

## Next Steps: Data Preprocessing
<!-- Update this section when data-preprocessing is compeleted -->
> **Note:** After completing `dataset-prep`, the next phase will be **data preprocessing**.  
> At this stage, the dataset is properly organized, verified, and split, but **no preprocessing** has been applied yet.  

Currently, there are only rough sketches of what data preprocessing will entail but the [issue #5](https://github.com/00015775/uzslr-isolated-dynamic/issues/5) is already open/created. Once the preprocessing scripts and workflow are finalized, this section will be updated and linked to the corresponding folder.  

---

# Mediapipe

**mediapipe version:** `0.10.21`

> **Note:** The MediaPipe section below is provided for documentation and reference only.  
> Its purpose is to explain how landmarks are structured, stored in `.npy` files, and reconstructed into their original shapes.  
> It is **not part of the `dataset-prep` pipeline**, and all `dataset-prep` steps are considered complete before this section.

---

## Landmark values

**Common for Face, Pose, and Hands:**
- **X**: Normalized by image width → ideally in `[0.0, 1.0]` range (left to right).
- **Y**: Normalized by image height → ideally in `[0.0, 1.0]` range (top to bottom).
In practice, values can slightly exceed `[0.0, 1.0]` (such as, >1.0 or <0.0) when the model extrapolates positions for partially occluded or out-of-frame body parts.

**Z Coordinate (Depth):**
- Relative depth (not absolute distance).
- Smaller (more negative) Z = closer to the camera.
- The magnitude/scale of Z is roughly the same as X (_uses a similar unit to the normalized image width_).
- Range: Typically around `-1.0 to +1.0` or a bit wider, depending on the component, but no strict bounds.

**Pose Visibility:**

MediaPipe's Pose Landmarker (including when used in the Holistic pipeline), the model always outputs all 33 pose landmarks with their `X, Y, Z` coordinates, even if a body part is occluded, partially visible, or completely outside the image frame.
The visibility field (a value between `0.0 and 1.0`) indicates the model's estimated likelihood that the landmark is visible (not occluded by another body part or object) within the frame.

- A high visibility (such as, close to 1.0) means the model is confident the point is directly observable.
- A low visibility (such as, close to 0.0) means it's likely occluded or not visible, but the model still provides a predicted position based on context from the rest of the body (using its learned human pose priors).

---

## Landmark `.npy` File Format

Each recorded frame is saved as a NumPy `.npy` file containing a **flattened vector of Mediapipe landmarks**.

One file corresponds to **one video frame**.

### File Structure

- **Path (After Data Splitting)**:  
  `data/{sign}/rep-{n}/frame-XX.npy`

- **Type**:  
  `numpy.ndarray`

- **Shape**:  
  `(1662,)`

- **Dtype**:  
  `float32` / `float64` (NumPy default)

- **Total `frame-*.npy` files**:  
  `82816`


## Vector Layout (Fixed Order)

The vector is a concatenation of four landmark groups, which are saved in a given order `return np.concatenate([face, pose, rh, lh])`:

### 1. Face Landmarks
- **468 points**
- **3 values per point**: `(x, y, z)`
- **Total values**: `468 × 3 = 1404`

### 2. Pose Landmarks
- **33 points**
- **4 values per point**: `(x, y, z, visibility)`
- **Total values**: `33 × 4 = 132`

> Even if some pose landmarks have `visibility = 0.0`, they are anyways saved into `.npy` and will be rendered in [`video-collector/dataset-checks/visualize_landmarks.py`](../video-collector/dataset-checks/04_visualize_landmarks.py).

### 3. Right Hand Landmarks
- **21 points**
- **3 values per point**: `(x, y, z)`
- **Total values**: `21 × 3 = 63`

### 4. Left Hand Landmarks
- **21 points**
- **3 values per point**: `(x, y, z)`
- **Total values**: `21 × 3 = 63`


## Total Vector Size

- 1404 (face)
- 132 (pose)
- 63 (right hand)
- 63 (left hand)
**= 1662 values**

<pre>
0                                                            1661
│──────────────────────────────────────────────────────────────│
│                                                              │
│  FACE (468×3)   POSE (33×4)   RIGHT HAND (21×3)  LEFT HAND   │
│  x y z ...      x y z v ...   x y z ...          x y z ...   │
│                                                              │
│  1404 values     132 values        63 values      63 values  │
│                                                              │
│───────────────┬─────────────┬────────────────┬───────────────│
0            1404          1536              1599          1662

v = visibility
</pre>


## Missing Landmarks

If a landmark group is **not detected** in a frame, its section is filled with **zeros** of the appropriate length with `np.zeros(A*B)`.
```python  
np.zeros(468*3) # face
np.zeros(33*4)  # pose
np.zeros(21*3)  # right hand
np.zeros(21*3)  # left hand
```

---

## Reconstructing Original Landmark Shapes

```python
import numpy as np

vec = np.load("frame-00.npy")

face = vec[0:1404].reshape(468, 3)
pose = vec[1404:1404+132].reshape(33, 4)
rh   = vec[1536:1536+63].reshape(21, 3)
lh   = vec[1599:1599+63].reshape(21, 3)
```

Alternatively (same result, no manual calculation needed):

```python
import numpy as np

vec = np.load("frame-00.npy")

face = vec[0:468*3].reshape((468, 3))
pose = vec[468*3:468*3 + 33*4].reshape((33, 4))
rh   = vec[468*3 + 33*4:468*3 + 33*4 + 21*3].reshape((21, 3))
lh   = vec[468*3 + 33*4 + 21*3:].reshape((21, 3))
```



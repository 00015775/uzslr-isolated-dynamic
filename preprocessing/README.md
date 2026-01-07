# Data Preprocessing Pipeline

> [!NOTE]
> This preprocessing pipeline is a **PyTorch adaptation** of the logical steps from Hoyso48 (2023) training notebook (author: Sohn, H., 2023).   
>
> The original notebook was written in TensorFlow and used a different dataset structure.  
> In this project, the pipeline is **not a direct copy** — it has been adapted for a **low-resource Uzbek Sign Language dataset** and PyTorch framework, with different folder organization and preprocessing steps suitable for this dataset.
>
> **Reference as to where this information was learned from, is provided below with relevant links.**

> [!IMPORTANT]
> **Workflow note:**  
> - Initial EDA was done in `01_ak_exploratory_analysis.ipynb`  
> - First preprocessing attempts were in `02_ak_preprocess_v1.ipynb`  
> - The current, final preprocessing and augmentation pipeline is in `02_ak_preprocess_v1.ipynb`  
> 
> **Naming convention (from Cookiecutter, n.d.):**  
> - `ak` = my initials  
> - `_vX` = version number, where higher means more refined improvements over the previous version (`_vX-1`)  
> - The leading numbers (`01`, `02`, etc.) indicate the **overall logical order** of the workflow steps

## Pipeline Stages

### Conda environment

```shell
# Create and activate the environment
conda env create -f environment-uzslr-signs.yml
conda activate uzslr-signs
```

> [!IMPORTANT]
> This environment, `uzslr-signs`, is currently configured to support all preprocessing steps.
> It will also be used for model training and evaluation once finalized.
> The [`environment-uzslr-signs.yml`](../environment-uzslr-signs.yml) will be expanded as new dependencies are added during the model development phase.


### Stage 1: Feature Selection

**Landmarks used:** Left and right hand, eye, nose, and lips landmarks (Sohn, 2023). Pose landmarks as per visualization analysis, did not provide useful information, that is why in this project pose landmarks are not considered.

> _Direct quote_: "I used left-right hand, eye, nose, and lips landmarks."  

Hence, for this project total of 118 landmarks out of 543 were chosen for this pipeline. 

> NOSE: 4   
> LIP: 40   
> REYE: 16   
> LEYE: 16   
> RHAND: 21   
> LHAND: 21   

<details>
<summary>
<b>To see the exact landmarks, click here:</b>
</summary>


This below given coordinates are taken from Sohn, H. (2023) training notebook, but have been modified to adapt to this project's dataset: https://www.kaggle.com/code/hoyso48/1st-place-solution-training?scriptVersionId=128283887&cellId=8


```python

NOSE=[
    1,2,98,327
]

LIP = [ 0, 
    61, 185, 40, 39, 37, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
]

REYE = [
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    246, 161, 160, 159, 158, 157, 173,
]
LEYE = [
    263, 249, 390, 373, 374, 380, 381, 382, 362,
    466, 388, 387, 386, 385, 384, 398,
]

RHAND = list(range(468 + 33, 468 + 33 + 21))      # [501, 502, ..., 522] 
LHAND = list(range(468 + 33 + 21, 468 + 33 + 42)) # [522, 523, ..., 542]

POINT_LANDMARKS = LIP + LHAND + RHAND + NOSE + REYE + LEYE # 118 landmarks in total

```

</details>

---

### Stage 2: Preprocessing (`Preprocess` Class)
The `Preprocess` class handles feature selection, normalization, and temporal feature engineering for each sample. It converts raw frame data into a fixed, standardized representation suitable for the model.

- **Firstly:** Unpack raw frame(s) `(1662,)` -> `(543, 3)`  
- **Then:** Extract 118 selected landmarks (hands, lips, eyes, nose), keep only `x, y` -> `(T, 118, 2)`  
- **Next:** Center and normalize landmarks relative to reference point (17th landmark) -> `(T, 118, 2)`  
- **Then:** Compute temporal dynamics: velocity and acceleration -> `(T, 118, 2)` each  
- **Finally:** Flatten and concatenate position, velocity, acceleration -> `(T, 708)` (final features)

---

### Stage 3: Data Augmentation

- **Firstly:** Temporal resampling to simulate varying gesture speeds -> may **increase or decrease number of frames** `(T, 543, 3)`  
- **Then:** Horizontal flip (mirror) of hands, lips, eyes, and nose with landmark swapping -> `(T, 543, 3)`  
- **Next:** Random spatial transformations (affine) including **scaling, rotation, shear, and shifts** -> `(T, 543, 3)`  
- **Then:** Temporal crop to limit sequence length -> `(MAX_LEN, 543, 3)`  
- **Finally:** Random spatial masking applied to a rectangular region across all frames -> `(MAX_LEN, 543, 3)`  

> Each augmentation is applied **randomly** with a predefined probability to increase variability in the training data.


---


### Stage 4: Dataset Loading (`SignDataset`)

- **Firstly:** Scan dataset directories to collect all repetitions and assign **integer labels** based on sign names.  
  - Each sample is stored as a list of `.npy` frame paths + label
- **Then:** Load all frames for a sample into memory and **convert to PyTorch tensors** `(T, 1662)`
- **Next:** **Unpack flat vectors** into structured landmarks: face, pose, left hand, right hand -> `(T, 543, 3)`
- **Then (optional):** Apply **augmentations** (`augment_fn`) which may modify coordinates and number of frames
- **Next:** Apply **`pad_or_truncate`** to ensure a **fixed sequence length** `(MAX_LEN, 543, 3)`
- **Then:** Apply **preprocessing** (`Preprocess`) for **feature selection, centering, normalization, and temporal dynamics** -> `(MAX_LEN, 708)`
- **Finally:** Return the **processed feature tensor** `x` and **label tensor** `y` for model training

---

### Stage 5: DataLoader

- **Firstly:** Wrap the `SignDataset` with PyTorch `DataLoader` to create **batches** for training  
- **Then:** Each batch has shape `(B, T, 708)` where:  
  - `B` = batch size  
  - `T` = number of frames per sequence (`MAX_LEN`)  
  - `708` = features per frame (position + velocity + acceleration for 118 landmarks)  
- **Finally:** The batch tensor is of type `torch.float32` and resides on `CPU` by default. Labels are returned as `torch.long` with `0-49` range


---


## Pipeline Summary

Complete preprocessing pipeline follows this sequence:

<p align="center">
  <img src="../docs/images/data_preprocess_augment_v1.png" alt="pipeline" width="450">
</p>

<details>
<summary><i>To see PlantUML code, click here.</i></summary>

## PlantUML code:

```code
@startuml
top to bottom direction

actor User

component "Disk: *.npy files\nOne file = one frame\nShape per file: (1662,)\ndtype: float64 (numpy)" as Disk

note right
  Each repetition rep-X folder,
  (stacked 32 .npy frames)
  is treated as a single training sample.
end note

component "Dataset.__getitem__\n\nnp.stack([...])\nShape: (T, 1662)\ndtype: float32 (numpy)" as GetItem

component "torch.from_numpy\nShape: (T, 1662)\ndtype: torch.float32\ndevice: CPU" as FromNumpy

note right
torch.float32 is chosen due to 
MPS acceleration compatibility
if used later, also takes less space
compared to torch.float64
end note

component "Vectorized UNPACK\n\nface: (T, 468, 3)\npose: (T, 33, 3)   ← visibility DROPPED\nrh  : (T, 21, 3)\nlh  : (T, 21, 3)\n\nconcat → (T, 543, 3)\nvalues: (x, y, z)" as Unpack

component "augment_fn\nInput: (T, 543, 3)\n\nresample        → T changes\nflip_lr         → x mirrored, hands swapped\ntemporal_crop   → T ≤ 32\nspatial_affine  → scale/shear/rotate/shift\nspatial_mask    → rectangular landmark dropout\n\nNo normalization here\nNo coordinate-range assumption\ndtype/device unchanged" as Augment

note right
temporal_mask is not included
due to possibility of 
introducing discountinities
end note

component "pad_or_truncate\nShape enforced: (32, 543, 3)\nPadding value: 0.0\ndtype: torch.float32\ndevice: CPU" as Pad

component "Preprocess.forward\nInput: (32, 543, 3)\n\n1) Landmark selection\n→ keep 118 landmarks\nShape: (32, 118, 3)\n\n2) Drop z coordinate\n→ keep (x, y)\nShape: (32, 118, 2)\n\n3) Center normalization\n→ reference landmark (nose)\n→ global over time\n\n4) Standard deviation normalization\n→ std over (T, N)\n\n5) Temporal derivatives\nvelocity     dx   : (32, 118, 2)\nacceleration dx2  : (32, 118, 2)\n\n6) Flatten per frame\nframes : (32, 236)\ndx     : (32, 236)\ndx2    : (32, 236)\n\nconcat → (32, 708)\ndtype: torch.float32\ndevice: CPU" as Preprocess

component "DataLoader batching\n\nx: (B, 32, 708)\ny: (B,)\ndtype: torch.float32 / torch.long\ndevice: CPU" as DataLoader

component "→ fed into model\n(model may later move tensors to GPU)" as Model

note right
Final tensors entering the model:
x:
  shape  = (B, 32, 708)
  dtype  = torch.float32
  device = CPU (until model.to(device))

y:
  shape  = (B,)
  dtype  = torch.long
  device = CPU
end note

User --> Disk
Disk --> GetItem
GetItem --> FromNumpy
FromNumpy --> Unpack
Unpack --> Augment
Augment --> Pad
Pad --> Preprocess
Preprocess --> DataLoader
DataLoader --> Model

@enduml

```
</details>


---

## Visualizing Frames
<!-- 
![right-hand](../docs/gifs/right_hand.gif)
![left-hand](../docs/gifs/left_hand.gif)
![both-hand](../docs/gifs/both_hand.gif)
![face](../docs/gifs/face.gif)
![full-body](../docs/gifs/full_body.gif)
![both-eyes](../docs/gifs/both_eyes.gif)
![lips](../docs/gifs/lip.gif)
![nose](../docs/gifs/nose.gif) -->

<table>
  <tr>
    <td align="center">
      <img src="../docs/gifs/right_hand.gif" alt="right-hand" width="300"><br>
      Right Hand
    </td>
    <td align="center">
      <img src="../docs/gifs/left_hand.gif" alt="left-hand" width="300"><br>
      Left Hand
    </td>
    <td align="center">
      <img src="../docs/gifs/both_hand.gif" alt="both-hand" width="300"><br>
      Both Hands
    </td>
  </tr>

  <tr>
    <td align="center">
      <img src="../docs/gifs/face.gif" alt="face" width="300"><br>
      Face
    </td>
    <td align="center">
      <img src="../docs/gifs/full_body.gif" alt="full-body" width="300"><br>
      Full Body
    </td>
    <td></td> 
  </tr>

  <tr>
    <td align="center">
      <img src="../docs/gifs/both_eyes.gif" alt="both-eyes" width="300"><br>
      Eyes
    </td>
    <td align="center">
      <img src="../docs/gifs/lip.gif" alt="lips" width="300"><br>
      Lips
    </td>
    <td align="center">
      <img src="../docs/gifs/nose.gif" alt="nose" width="300"><br>
      Nose
    </td>
  </tr>
</table>

---

**References List**
-

Bergeron, M. (2024). Insightful Datasets for ASL recognition. Hackster.io. Available at: https://www.hackster.io/AlbertaBeef/insightful-datasets-for-asl-recognition-f786b9 [Accessed: 28 December 2025]

Computer Vision Engineer. (2023). _Sign Language Detection with Python and Scikit Learn | Landmark Detection | Computer Vision Tutorial_. [Video] Available at: https://www.youtube.com/watch?v=MJCSjXepaAM&t=3148s [Accessed: 27 October 2025]

Cookiecutter (n.d.). _Using the template – Cookiecutter Data Science_. Available at: https://cookiecutter-data-science.drivendata.org/using-the-template/ (Accessed: 2 January 2026)

Goncharov, I. (2022). _Custom Hand Gesture Recognition with Hand Landmarks Using Google’s Mediapipe + OpenCV in Python_. [Video] Available at: https://www.youtube.com/watch?v=a99p_fAr6e4&list=PL0FM467k5KSyt5o3ro2fyQGt-6zRkHXRv [Accessed: 27 October 2025]

Hoyso48 (2023). _1st place solution ‑ training [Kaggle notebook]_. Available at: https://www.kaggle.com/code/hoyso48/1st-place-solution-training?scriptVersionId=128283887&cellId=8 (Accessed: 27 December 2025)

Renotte, N. (2021). _Sign Language Detection using ACTION RECOGNITION with Python | LSTM Deep Learning Model_. [Video] Available at: https://www.youtube.com/watch?v=doDUihpj6ro [Accessed: 21 October 2025]

Sohn, H. (2023). _1st place solution - 1DCNN combined with Transformer_. Available at: https://www.kaggle.com/competitions/asl-signs/writeups/hoyeol-sohn-1st-place-solution-1dcnn-combined-with [Accessed: 27 December 2025]




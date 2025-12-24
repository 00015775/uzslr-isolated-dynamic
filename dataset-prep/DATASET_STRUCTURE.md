# Dataset Structure Blueprint

## Original Structure

This folder structure is generated during the `video-collector` phase of the project. While it is designed to be well-organized and easy to manage, it is not optimal for direct use in model training. The primary reason is that the main input features consist of individual `frame-XX.npy` files and not the videos at all, so video-related folders are eliminated in the subsequent designs. Additionally, the dataset needs to be split according to specific ratios, with folders and files reorganized into the appropriate `train/validation/test` structure, as detailed in the sections that will follow.

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

## Initial (Pre-Splitting) Structure
<pre>
.
└── data/
    └── {sign}/                      # e.g., assalomu_alaykum, bahor, birga, ...
         └── rep-{XX}/            # e.g., rep-0, rep-1, ..., rep-XX (repetitions)
             ├── frame-00.npy
             ├── frame-01.npy
             ├── ... 
             └── frame-XX.npy
</pre>

## Post-Splitting Structure

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

## Dataset Splitting Strategy

The dataset will be split into three subsets: **training**, **validation**, and **test**, using the following ratios:

- **Training**: 80%
- **Validation**: 10%
- **Test**: 10%

#### Key Requirements

- **Stratification by sign**: The split must be stratified at the level of unique signs. This ensures that the distribution of signs remains consistent across all three splits.
- **Presence in all splits**: Every unique sign must appear in the training, validation, and test sets. No sign should be entirely excluded from any split.

#### Splitting Mechanism

- The splitting is performed based on the individual repetition folders (`rep-XX`).
- For each unique sign:
  - All available `rep-XX` folders (repetitions) are collected.
  - These repetitions are randomly shuffled.
  - 80% of the repetitions are assigned to the **training** split.
  - 10% of the repetitions are assigned to the **validation** split.
  - The remaining 10% of the repetitions are assigned to the **test** split.
- Entire `rep-XX` folders (with all their contents) are moved or copied to the corresponding split directory without further subdivision.


## Other Dataset Info
- **Total signs:** 50
- **Repetitions per sign:** 38 signs × 52 repetitions, 12 signs × 51 repetitions
- **File format:** NumPy arrays (`.npy`)
- **Glossary:** "rep" = repetition
- **Total repetition folders:** 2588 (38x52 + 12x51)
- **Total `frame-XX.npy` files:** 82816

**Reference** 
Information learned from [HuggingFace - File names and splits](https://huggingface.co/docs/hub/datasets-file-names-and-splits)

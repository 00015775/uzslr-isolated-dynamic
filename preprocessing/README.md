# Data Preprocessing Strategy for UzSLR

> [!NOTE]
> This `README` is intentionally verbose to document the full preprocessing rationale and pipeline.  
> This heading mainly states the problem. If you want to skip it to data preprocessing pipeline, [click here](#data-preprocessing-pipeline).  
> **Reference as to where this information was learned from, is provided below with relevant links.**

## Problem Statement

Without proper preprocessing (normalization/alignment), our model would treat the same sign performed at different positions in the frame as entirely different patterns. This occurs because raw pixel values or landmark coordinates are position-dependent, meaning a sign like "_yaxshi_" performed in different spatial locations would generate different numerical inputs despite being semantically identical.

## Root Cause

When working with hand landmarks, the output consists of coordinate sets:
<pre>
(x₁,y₁), (x₂,y₂), ...
</pre>

With absolute pixel positions:
- Hand movement within the frame alters all coordinate values
- The feature space transforms accordingly
- The model interprets these as distinct samples

## Learning Objectives

### What the Model Should Learn
- Shape and motion patterns that define sign meaning
- Temporal changes in hand configuration (semantic content)
- Relative geometry of landmarks (finger and joint relationships)

### What the Model Should NOT Learn
- Hand position within the camera frame
- Apparent hand size (camera distance effects)
- Hand rotation or tilt in the image
- Signing speed or tempo

These factors vary across individuals and recordings but don't affect the linguistic meaning of signs.

## Impact of Preprocessing

### Without Preprocessing
The model might incorrectly learn:
> "Coordinates near (0.3, 0.5) represent '_yaxshi_'"

When the same sign appears near (0.7, 0.2), the differing raw values cause the model to misclassify it as a different gesture.

### With Preprocessing
By applying translation, scaling, and rotation normalization, we ensure every hand appears:
- Centered at the origin (translation invariance)
- With consistent size (scale invariance)
- In the same orientation (rotation invariance)

The model now focuses exclusively on spatial relationships between landmarks - the actual hand shape.

## Temporal Normalization

Consider a dynamic sign like "_boshlanishi_":
- **Signer A**: Completes in 3 seconds (15 frames)
- **Signer B**: Completes in 6 seconds (30 frames)

### Without Temporal Normalization
- Sequences have different lengths
- The model may incorrectly interpret speed as semantic information
- Incorrect learning: "Faster motion = Class A, slower motion = Class B"

### With Temporal Normalization
We remove speed as a variable, teaching the model that tempo doesn't alter meaning.

## Feature Engineering Principles

We design features to be:
- **Invariant** to irrelevant differences
- **Discriminative** for meaningful distinctions

| Feature Type | Invariant To | Sensitive To |
|-------------|--------------|--------------|
| Spatial | Position, scale, rotation | Hand shape, finger arrangement |
| Temporal | Speed, duration | Motion trajectory, direction, rhythm |

## Expected Outcomes

Through these preprocessing steps, we implicitly instruct the model:
> "Don't memorize spatial or temporal appearance, but rather learn the underlying pattern based on relative geometry and motion."

**Results**:
- Signs performed in different frame locations → identical features
- Signs performed at different speeds → consistent temporal patterns after temporal normalization
- Different signs (distinct motion paths/shapes) → distinguishable feature sequences

This approach enables our model to generalize robustly across different signers, and performance tempos.

## Analogy

For example, consider learning a dance move. We don't focus on where on stage it's performed or the dancer's size. We recognize it through the sequence of poses and transitions, basically its shape through time. Our model operates similarly, learning relative configuration and motion rather than absolute position or speed.

---

# Data Preprocessing Pipeline

## Pipeline Stages


### Stage 1: Feature Selection

**Landmarks used:** Left and right hand, eye, nose, and lips landmarks (Sohn, 2023). Since pose landmarks are not listed, it could have been disregarded in order to reduce noise from full-body pose, since all 33 landmarks are provided, even if none is detected.

> _Direct quote_: "I used left-right hand, eye, nose, and lips landmarks."  

As per the review done by Bergeron, M. (2024), the top winner used the 130 landmarks:
> [!IMPORTANT]
> The input to the solution are the following subset of 130 landmarks:
>
> 21 - key points from each hand
> 6 -  pose key points from each arm
> 76 - from the face (lips, nose, eyes)

<details>
<summary>
<b>To see the exact landmarks, click here:</b>
</summary>


This below given coordinates are taken from Sohn, H. (2023) training notebook: https://www.kaggle.com/code/hoyso48/1st-place-solution-training?scriptVersionId=128283887&cellId=8


```python
NOSE=[
    1,2,98,327
]
LNOSE = [98]
RNOSE = [327]
LIP = [ 0, 
    61, 185, 40, 39, 37, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
]
LLIP = [84,181,91,146,61,185,40,39,37,87,178,88,95,78,191,80,81,82]
RLIP = [314,405,321,375,291,409,270,269,267,317,402,318,324,308,415,310,311,312]

POSE = [500, 502, 504, 501, 503, 505, 512, 513]
LPOSE = [513,505,503,501]
RPOSE = [512,504,502,500]

REYE = [
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    246, 161, 160, 159, 158, 157, 173,
]
LEYE = [
    263, 249, 390, 373, 374, 380, 381, 382, 362,
    466, 388, 387, 386, 385, 384, 398,
]

LHAND = np.arange(468, 489).tolist()
RHAND = np.arange(522, 543).tolist()
```

</details>

---

### Stage 2: Dataset Splitting

**Timing**: Before any normalization or augmentation

We split our dataset into three subsets:
- **Training Set**: For model learning
- **Validation Set**: For hyperparameter tuning and model selection
- **Test Set**: For final performance evaluation

**Rationale**: This separation must occur first to prevent data leakage and ensure proper evaluation of our model's generalization capability.

---

### Stage 3: Geometric Normalization

**Scope**: Applied per frame  
**Purpose**: Remove irrelevant spatial differences between signers

We apply the following transformations in sequence:

#### 3.1 Translation Invariance
- Compute the mean position of the 17th landmark (nose) across all frames in the sequence.  
- If this mean contains `NaN` values, replace them with 0.5.  
- Subtract this mean position from all selected landmark coordinates (in every frame).  

![normalization-reference-point](../docs/images/norm_reference_point(sohn-h).png)

Kaggle (2023)._1st Place Solution: Google Isolated Sign Language Recognition with Hoyeol Sohn | Kaggle_. Available from: https://youtu.be/eH-z0b6lfxs?list=PLqFaTIg4myu-703qAHB7yHKUIKHxazMjY&t=252 (Accessed: 28 December 2025)

**Effect:** Centers the sequence around the average nose position, making features position-independent (Sohn, 2023).

> _Direct reference_: "For normalization, I used the 17th landmark located in the nose as a reference point, since it is usually located close to the center ([0.5, 0.5])."

#### 3.2 Scale Invariance
- After centering, compute the standard deviation of the landmark coordinates (across all frames, landmarks, and x/y/z dimensions, using the centered values).  
- Divide all centered coordinates by this standard deviation.  

**Effect:** Normalizes the sequence to unit variance, reducing sensitivity to overall scale differences due to camera distance or signer size (Sohn, 2023).

---

### Stage 4: Temporal Alignment

> [!NOTE]
> **Status**: NOT REQUIRED — Already handled by `video-collector` phase  
> **Scope**: Applied per video/sequence  
> **Purpose**: Handle speed and duration differences

If needed in future iterations, we would:
- Resample every sign to a fixed number of frames (such as, 32 or 64)

**Output**: All sequences with equal temporal length for direct comparison

---

### Stage 5: Numerical Normalization

**Scope**: Training set only  
**Purpose**: Ensure feature values have consistent scales for stable training

We apply one of the following methods across the entire training dataset:

| Method | Formula | Use Case |
|--------|---------|----------|
| **Z-score (Standardization)** | `x' = (x - μ) / σ` | Center around 0, scale by variance |
| **Min-Max Scaling** | `x' = (x - x_min) / (x_max - x_min)` | scales to [0,1] |

#### Critical Implementation Details

**Statistics Computation**:
- Calculate mean (μ), standard deviation (σ), min, and max values **only** on the original, non-augmented training dataset
- **Do not** include validation or test sets in these calculations
- **Do not** include augmented samples in these calculations

**Application**:
- Fit normalization parameters on training set
- Apply the same transformation to validation and test sets using training statistics

**Output**: Final normalized numeric feature tensors ready for model input

> [!IMPORTANT]
> #### Processing Order:
> 1. **First**: Apply geometric normalization (invariance), which helps to define what the model should ignore (position, size, rotation)   
> 2. **Second**: Apply numerical scaling for consistent training behavior, which helps to optimize how the model processes data (consistent magnitude)


---

### Stage 6: Dimensionality Reduction (Optional)

> [!NOTE]
> **Status**: NOT CONSIDERED for current implementation  
> **Future Consideration**: PCA (Principal Component Analysis)

**Important**: 
- Fit PCA on training set only
- Transform validation and test sets using training-fitted parameters

---

### Stage 7: Data Augmentation

**Scope**: Training set only, applied on-the-fly during training  
**Purpose**: Increase training data diversity and model robustness  

The following data augmentation strategies were employed in the winning solution for the [Google - Isolated Sign Language Recognition Kaggle competition](https://www.kaggle.com/competitions/asl-signs/overview), which utilized a 1D CNN-based model on landmark sequences (Sohn, 2023).

#### Temporal Augmentation  
_These augmentations modify the time dimension of the sequences to account for variations in signing speed and potential occlusions or noise._  

1. **Random Resample (0.5x ~ 1.5x to original length)**  
> Randomly adjusts the sequence length by resampling with a speed factor between 0.5 and 1.5, using linear interpolation to create longer (slower) or shorter (faster) sequences. This promotes tempo invariance in sign recognition.  
2. **Random Masking**  
> Randomly masks (such as, zeros out) portions of frames in the sequence, simulating occlusions or landmark detection failures and encouraging the model to use temporal context.  

#### Spatial Augmentation  
_These augmentations transform landmark coordinates to handle variations in pose, position, and orientation._  

3. **Horizontal Flip**  
> Mirrors all x-coordinates across the vertical axis for the entire sequence, helping in robustness to left/right-handed variations and camera angles.  
4. **Random Affine (Scale, Shift, Rotate, Shear)**  
> Applies a consistent random affine transformation across all frames, including scaling, translation, rotation, and shearing, to simulate differences in body scale, position, and tilt.  
5. **Random Cutout**  
> Randomly occludes regions of landmarks (such as, by setting to zero), mimicking partial body part occlusions or detection errors.  

These augmentations significantly contributed to the model's generalization and top performance.  

**Important**: 
- **No augmentation** is applied to validation or test sets  
- Augmentation occurs after normalization  
- Applied dynamically during training iterations  

---

## Pipeline Summary

Our complete preprocessing pipeline follows this sequence:

<pre>
1. Feature Selection (left/right hand, eyes, nose, lips landmarks)
   ↓
2. Dataset Split (Train/Val/Test)
   ↓
3. Geometric Normalization (Translation → Scale → Rotation)
   → Removes irrelevant camera and spatial differences
   ↓
4. Temporal Alignment (SKIPPED - handled by video-collector)
   → Would remove signing-speed differences
   ↓
5. Numerical Normalization (Z-score or Min-Max)
   → Stabilizes numeric scale for training
   ↓
6. Dimensionality Reduction (OPTIONAL - not implemented)
   → Would remove redundancy/noise and compress information
   ↓
7. Data Augmentation (Training only, on-the-fly)
   → Increases data diversity
   ↓
8. Model Training
   → Learns mapping from motion pattern to sign label
</pre>

---

## Key Principles

### Data Leakage Prevention
- All preprocessing statistics (normalization parameters, PCA components) are computed exclusively on the training set
- The same preprocessing transformations are applied to validation and test sets using training-derived parameters
- Augmentation is never applied to validation or test data

### Invariance Hierarchy
1. **Geometric invariance**: Spatial consistency across signers
2. **Temporal invariance**: Speed-independent representations
3. **Numerical invariance**: Scale-consistent features for optimization

### Processing Order Rationale
- **Split first**: Prevents data leakage
- **Geometric before numerical**: Shape normalization precedes value scaling
- **Augmentation last**: Applied dynamically to normalized data during training

---

## Expected Outcomes

By following this pipeline, we ensure that:
- The model learns sign semantics, not spatial artifacts
- Training is numerically stable and efficient
- Evaluation metrics reflect true generalization performance
- The system is robust to variations in signer position, size, and camera setup

---

**References List**
-

Bergeron, M. (2024) Insightful Datasets for ASL recognition. Hackster.io. Available at: https://www.hackster.io/AlbertaBeef/insightful-datasets-for-asl-recognition-f786b9 [Accessed: 28 December 2025]

Computer Vision Engineer. (2023). _Sign Language Detection with Python and Scikit Learn | Landmark Detection | Computer Vision Tutorial_. [Video] Available at: https://www.youtube.com/watch?v=MJCSjXepaAM&t=3148s [Accessed: 27 October 2025]

Goncharov, I. (2022). _Custom Hand Gesture Recognition with Hand Landmarks Using Google’s Mediapipe + OpenCV in Python_. [Video] Available at: https://www.youtube.com/watch?v=a99p_fAr6e4&list=PL0FM467k5KSyt5o3ro2fyQGt-6zRkHXRv [Accessed: 27 October 2025]

Renotte, N. (2021). _Sign Language Detection using ACTION RECOGNITION with Python | LSTM Deep Learning Model_. [Video] Available at: https://www.youtube.com/watch?v=doDUihpj6ro [Accessed: 21 October 2025]

Sohn, H. (2023). _1st place solution - 1DCNN combined with Transformer_. Available at: https://www.kaggle.com/competitions/asl-signs/writeups/hoyeol-sohn-1st-place-solution-1dcnn-combined-with [Accessed: 27 December 2025]


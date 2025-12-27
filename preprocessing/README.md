# Data Preprocessing Strategy for UzSLR

> [!NOTE]
> This `README` is intentionally verbose to document the full preprocessing rationale and pipeline.
> Reference as to where this information was learned from, is provided below with relevant link.

## Problem Statement

Without proper preprocessing (normalization/alignment), our model would treat the same sign performed at different positions in the frame as entirely different patterns. This occurs because raw pixel values or landmark coordinates are position-dependent, meaning a sign like "_yaxshi_" performed in different spatial locations would generate different numerical inputs despite being semantically identical.

## Root Cause Analysis

When working with hand landmarks, the output consists of coordinate sets:
<pre>
(x₁,y₁), (x₂,y₂), ...
</pre>

With absolute pixel positions:
- Hand movement within the frame alters all coordinate values
- The feature space transforms accordingly
- The model interprets these as distinct samples

## Proposed Solution: Geometric Normalization

We need to establish representation invariance to translation, scale, and rotation. Our approach includes several normalization methods:

### 1. Translation Normalization (Relative to Reference point)
We make all coordinates relative to the wrist or palm center:
<pre>
(x'ᵢ, y'ᵢ) = (xᵢ - x_wrist, yᵢ - y_wrist)
</pre>
**Effect**: Removes position dependence

### 2. Scale Normalization
We divide all coordinates by a reference distance (such as, wrist to middle fingertip):
<pre>
(x''ᵢ, y''ᵢ) = (x'ᵢ, y'ᵢ) / d_ref
</pre>
**Effect**: Removes scale differences caused by varying hand-to-camera distances

### 3. Rotation Normalization (Optional)
We rotate all points to ensure the wrist-to-index direction maintains consistent orientation.

**Effect**: Handles variations in hand tilt

**Result**: The same sign performed at different locations is now treated as the same pattern.

## Distinction: Geometric vs. Numerical Normalization

### Geometric Normalization (Invariance)
This process removes unwanted spatial variations but doesn't constrain coordinate values to specific numerical ranges. After geometric normalization:
- Coordinates might range around [-0.5, 0.8]
- Scale variations may persist based on proportions
- Rotation preserves the original range

**Purpose**: Numerical normalization

### Feature Normalization (Range Scaling)
After establishing geometric invariance, we apply numerical normalization to stabilize training:

| Method | Formula | Use Case |
|--------|---------|----------|
| Min-Max Scaling | `x' = (x - x_min) / (x_max - x_min)` | Squeeze into [0,1] |
| Z-score (Standardization) | `x' = (x - μ) / σ` | Center around 0, scale by variance |

### Processing Order
1. **First**: Apply geometric normalization (invariance)
2. **Second**: Apply numerical scaling for consistent training behavior

### Conceptual Framework
- **Geometric normalization** → Defines what the model should ignore (position, size, rotation)
- **Numeric normalization** → Optimizes how the model processes data (consistent magnitude)

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

### Stage 1: Dataset Splitting

**Timing**: Before any normalization or augmentation

We split our dataset into three subsets:
- **Training Set**: For model learning
- **Validation Set**: For hyperparameter tuning and model selection
- **Test Set**: For final performance evaluation

**Rationale**: This separation must occur first to prevent data leakage and ensure proper evaluation of our model's generalization capability.

---

### Stage 2: Geometric Normalization

**Scope**: Applied per frame  
**Purpose**: Remove irrelevant spatial differences between signers

We apply the following transformations in sequence:

#### 2.1 Translation Invariance
We subtract a reference point from all coordinates:
- For upper body data: Use `mid_shoulders` as reference
- For hand-only data: Use `wrist` as reference

**Effect**: All landmarks become position-independent

#### 2.2 Scale Invariance
We normalize all coordinates by dividing by a reference distance:
- Shoulder width for full body
- Hand length for hand-only data

**Effect**: Removes size variations due to different camera distances or physical differences between signers

#### 2.3 Rotation Invariance (Optional)
We rotate all points to establish a consistent orientation:
- Align shoulders horizontally for upper body
- Align hand base horizontally for hand-only data

**Output**: "Aligned" landmark coordinates where all signs share the same reference frame

---

### Stage 3: Temporal Alignment

> [!NOTE]
> **Status**: NOT REQUIRED — Already handled by `video-collector` phase  
> **Scope**: Applied per video/sequence  
> **Purpose**: Handle speed and duration differences

If needed in future iterations, we would:
- Resample every sign to a fixed number of frames (such as, 32 or 64)
- Use interpolation for frame generation

**Output**: All sequences with equal temporal length for direct comparison

---

### Stage 4: Numerical Normalization

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

---

### Stage 5: Dimensionality Reduction (Optional)

> [!NOTE]
> **Status**: NOT CONSIDERED for current implementation  
> **Future Consideration**: PCA (Principal Component Analysis)

**Important**: 
- Fit PCA on training set only
- Transform validation and test sets using training-fitted parameters

---

### Stage 6: Data Augmentation

**Scope**: Training set only, applied on-the-fly during training  
**Purpose**: Increase training data diversity and model robustness

We apply the following augmentation techniques randomly:
- Horizontal flip
- Rotation
- ... etc. 

**Important**: 
- **No augmentation** is applied to validation or test sets
- Augmentation occurs after normalization
- Applied dynamically during training iterations

---

## Pipeline Summary

Our complete preprocessing pipeline follows this sequence:

<pre>
1. Dataset Split (Train/Val/Test)
   ↓
2. Geometric Normalization (Translation → Scale → Rotation)
   → Removes irrelevant camera and spatial differences
   ↓
3. Temporal Alignment (SKIPPED - handled by video-collector)
   → Would remove signing-speed differences
   ↓
4. Numerical Normalization (Z-score or Min-Max)
   → Stabilizes numeric scale for training
   ↓
5. Dimensionality Reduction (OPTIONAL - not implemented)
   → Would remove redundancy/noise and compress information
   ↓
6. Data Augmentation (Training only, on-the-fly)
   → Increases data diversity
   ↓
7. Model Training
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
Computer Vision Engineer. (2023). Sign Language Detection with Python and Scikit Learn | Landmark Detection | Computer Vision Tutorial. [Video] Available at: https://www.youtube.com/watch?v=MJCSjXepaAM&t=3148s [Accessed: 27 October 2025]

Goncharov, I. (2022). Custom Hand Gesture Recognition with Hand Landmarks Using Google’s Mediapipe + OpenCV in Python. [Video] Available at: https://www.youtube.com/watch?v=a99p_fAr6e4&list=PL0FM467k5KSyt5o3ro2fyQGt-6zRkHXRv [Accessed: 27 October 2025]

Renotte, N. (2021). Sign Language Detection using ACTION RECOGNITION with Python | LSTM Deep Learning Model. [Video] Available at: https://www.youtube.com/watch?v=doDUihpj6ro [Accessed: 21 October 2025]



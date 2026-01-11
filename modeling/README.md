# Model Development

## Conda environment

This model code uses the same conda environment as [`preprocessing`](../preprocessing/).

```shell
# create and activate the environment
conda env create -f environment-uzslr-signs.yml
conda activate uzslr-signs
```

## Model Architecture

Hybrid CNN-Transformer architecture for sign language recognition.

### Input Shape
- **Raw input**: `(batch, 32, 1662)` - 32 frames of MediaPipe landmarks
- **After preprocessing**: `(batch, 32, 708)` - selected landmarks with temporal features

### Output Shape
- **Logits**: `(batch, 50)` - predictions for 50 Uzbek sign classes

### Architecture Overview

```
Input (32, 708)
    ↓
Stem: Linear + BatchNorm
    ↓
Group 1: 3× Conv1DBlock + 1× TransformerBlock
    ↓
Group 2: 3× Conv1DBlock + 1× TransformerBlock
    ↓
[Optional: Groups 3-4 for dim=384]
    ↓
Global Average Pooling + Classifier
    ↓
Output (50 classes)
```

## Model Configurations

- **Base model**: `dim=192` (2 groups, ~2.5M parameters)
- **Large model**: `dim=384` (4 groups, ~10M parameters)

## Key Components

- **Conv1DBlock**: causal depthwise convolution with ECA attention
- **TransformerBlock**: multi-head self-attention with FFN
- **Preprocessing**: landmark selection, normalization, velocity, acceleration

## Feature Engineering

From 543 MediaPipe landmarks:
- Select 118 key landmarks (face, hands, eyes, lips, nose)
- Extract (x, y) coordinates only
- Compute velocity (1st difference)
- Compute acceleration (2nd difference)
- Result: 118 landmarks × 2 coords × 3 features = 708 dimensions

## Training

- **Loss**: CrossEntropyLoss
- **Optimizer**: AdamW (lr=5e-4, weight_decay=0.1)
- **Batch size**: 16
- **Max epochs**: 300 with early stopping (patience=15)
- **Augmentation**: flip, resample, affine, spatial mask

## Outputs

- `best_model.pth`: model with best validation accuracy
- `checkpoint.pth`: latest training checkpoint (for resuming)

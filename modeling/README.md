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

- **Base model**: `dim=192` (2 groups, 1,753,040 or ~1,75M parameters)
- **Large model**: `dim=384` (4 groups, ~8-10M parameters)

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

## Classification Report

<pre>
precision    recall  f1-score   support

           0       0.86      1.00      0.92         6
           1       1.00      0.83      0.91         6
           2       1.00      1.00      1.00         6
           3       1.00      1.00      1.00         6
           4       1.00      1.00      1.00         6
           5       1.00      1.00      1.00         6
           6       1.00      0.33      0.50         6
           7       1.00      0.83      0.91         6
           8       0.67      1.00      0.80         6
           9       0.62      0.83      0.71         6
          10       1.00      0.17      0.29         6
          11       1.00      0.83      0.91         6
          12       1.00      1.00      1.00         6
          13       1.00      1.00      1.00         6
          14       0.60      1.00      0.75         6
          15       1.00      1.00      1.00         6
          16       0.67      1.00      0.80         6
          17       1.00      1.00      1.00         6
          18       0.80      0.67      0.73         6
          19       1.00      1.00      1.00         6
          20       0.67      0.67      0.67         6
          21       0.67      0.67      0.67         6
          22       0.67      0.67      0.67         6
          23       1.00      1.00      1.00         6
          24       0.83      0.83      0.83         6
          25       1.00      1.00      1.00         6
          26       0.86      1.00      0.92         6
          27       1.00      0.83      0.91         6
          28       1.00      0.83      0.91         6
          29       1.00      0.83      0.91         6
          30       1.00      0.50      0.67         6
          31       0.75      1.00      0.86         6
          32       1.00      1.00      1.00         6
          33       1.00      1.00      1.00         6
          34       1.00      0.67      0.80         6
          35       0.86      1.00      0.92         6
          36       1.00      1.00      1.00         6
          37       1.00      1.00      1.00         6
          38       1.00      1.00      1.00         6
          39       1.00      0.67      0.80         6
          40       0.55      1.00      0.71         6
          41       0.86      1.00      0.92         6
          42       0.83      0.83      0.83         6
          43       0.75      1.00      0.86         6
          44       1.00      0.67      0.80         6
          45       0.67      0.67      0.67         6
          46       1.00      0.83      0.91         6
          47       1.00      1.00      1.00         6
          48       1.00      1.00      1.00         6
          49       0.86      1.00      0.92         6

    accuracy                           0.87       300
   macro avg       0.90      0.87      0.87       300
weighted avg       0.90      0.87      0.87       300
</pre>

### Worst Classes

<pre>
WORST CLASSES:

internet              f1=0.286  p=1.000  r=0.167
bozor                 f1=0.500  p=1.000  r=0.333
mehmonxona            f1=0.667  p=0.667  r=0.667
mehribon              f1=0.667  p=0.667  r=0.667
metro                 f1=0.667  p=0.667  r=0.667
poezd                 f1=0.667  p=1.000  r=0.500
toza                  f1=0.667  p=0.667  r=0.667
shokolad              f1=0.706  p=0.545  r=1.000
iltimos               f1=0.714  p=0.625  r=0.833
likopcha              f1=0.727  p=0.800  r=0.667
</pre>

### Best Classes
<pre>
BEST CLASSES:

yopish                f1=1.000  p=1.000  r=1.000
yomg'ir               f1=1.000  p=1.000  r=1.000
restoran              f1=1.000  p=1.000  r=1.000
quyon                 f1=1.000  p=1.000  r=1.000
qorong'i              f1=1.000  p=1.000  r=1.000
qish                  f1=1.000  p=1.000  r=1.000
qidirish              f1=1.000  p=1.000  r=1.000
o'ynash               f1=1.000  p=1.000  r=1.000
musiqa                f1=1.000  p=1.000  r=1.000
maktab                f1=1.000  p=1.000  r=1.000
</pre>


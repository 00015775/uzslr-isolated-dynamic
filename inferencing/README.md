# Real-Time UzSLR Inference

Real-time inference system for Uzbek Sign Language recognition using a trained PyTorch model.

## Requirements

This inference system uses the same conda environment as [`preprocessing`](../preprocessing/) and [`modeling`](../modeling/).

### Conda environment
```shell
# create and activate the environment
conda env create -f environment-uzslr-signs.yml
conda activate uzslr-signs
```

### Key dependencies
- `mediapipe==0.10.21`
- `opencv==4.12.0`
- `pytorch==2.5.1`
- `numpy==2.0.2`

## File Structure

```
.
└──inferencing/
    ├── inference01_config.py      # configuration settings
    ├── inference02_preprocess.py  # preprocessing pipeline
    ├── inference03_model.py       # model architecture
    ├── inference04_main.py        # main inference script
    ├── best_model.pth            # trained model weights
    └── README.md                 
```

## Setup

1. Copy trained model to this directory:
   ```bash
   cp ../modeling/notebooks/best_model.pth .
   ```

2. Verify configuration in [`inference01_config.py`](./inference01_config.py):
   - Set `MODEL_DIM` to match your training (192 or 384)
   - Adjust `VIDEO_DEVICE` if needed (0 for default camera)

## Usage

```bash
# must be run, inside of inferencing folder
cd inferencing

python inference04_main.py
```

### Controls
- **q**: quit application

## How It Works

1. **Hand detection**: waits for both hands to be visible
2. **Frame collection**: automatically collects 32 frames
3. **Inference**: runs prediction every 0.5 seconds
4. **Display**: shows predicted sign name and confidence at top of screen

## Features

- automatic inference (no button pressing)
- requires both hands visible to start
- tolerates brief hand disappearance (5 frames but can be adjusted from [`inference01_config.py`](./inference01_config.py))
- real-time visualization with landmarks
- cross-platform device support (MPS/CUDA/CPU)

## Troubleshooting

**Camera not opening:**
- check `VIDEO_DEVICE` in config (try 0, 1, or 2)

**Model not loading:**
- verify `best_model.pth` exists in this directory
- check `MODEL_DIM` matches training configuration

**Performance:**
- uses MPS on Apple Silicon
- uses CUDA on NVIDIA GPUs
- falls back to CPU otherwise

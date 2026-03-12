# Sign Language Production Pipeline

> [!NOTE]
> The code in this folder was originally executed on an Ubuntu server. The folder is kept only as a reference for how to extract [`dwpose`](https://github.com/IDEA-Research/DWPose) coordinates and generate videos using [`musepose`](https://github.com/TMElyralab/MusePose). Because later updates were not applied here, the code may no longer work or may be outdated.

Generates pre-rendered avatar sign videos from signer10 source videos.
Run Phase 1 locally (macOS), Phase 2 on the Ubuntu RTX 5090 machine.

---

## Phase 1 — DWPose extraction (your Mac, no GPU needed)

### Install dependencies (once)
```bash
pip install opencv-python numpy tqdm onnxruntime huggingface_hub
```

### Test on 3 signs first
```bash
cd generative
```

```bash
python step01_extract_dwpose.py \
    --dataset_root ../Data_Numpy_Arrays_RSL_UzSL \
    --output_dir   ./dwpose_output \
    --signs maktab kitob qidirish
```

Check `./dwpose_output/maktab/overlay/` — you should see 32 PNG files
with white skeleton drawn on black background.

### Run on all 50 signs
```bash
python step01_extract_dwpose.py \
    --dataset_root ../Data_Numpy_Arrays_RSL_UzSL \
    --output_dir   ./dwpose_output \
```

### Transfer to Ubuntu
```bash
# via scp:
scp -r ./dwpose_output user@ubuntu-machine:/path/to/sign_production/

# or zip and copy via USB:
zip -r dwpose_output.zip dwpose_output/
```

Output size estimate: ~50MB (50 signs × 32 frames × ~30KB per PNG)

---

## Phase 2 — MusePose inference (Ubuntu, RTX 5090)

### Setup MusePose (once)
```bash
git clone https://github.com/TMElyralab/MusePose
cd MusePose

conda create -n musepose python=3.10 -y
conda activate musepose

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Download model weights (once, ~8GB total)
```bash
# inside MusePose directory:
python -c "
from huggingface_hub import snapshot_download
snapshot_download('TMElyralab/MusePose',
                  local_dir='./pretrained_weights/MusePose')
snapshot_download('runwayml/stable-diffusion-v1-5',
                  local_dir='./pretrained_weights/stable-diffusion-v1-5')
"

python -c "
from huggingface_hub import hf_hub_download
import os; os.makedirs('./pretrained_weights/dwpose', exist_ok=True)
hf_hub_download('yzd-v/DWPose', 'dw-ll_ucoco_384.onnx',
                local_dir='./pretrained_weights/dwpose')
hf_hub_download('yzd-v/DWPose', 'yolox_l.onnx',
                local_dir='./pretrained_weights/dwpose')
"
```

### Get a reference image
Use one of the MusePose sample images for initial testing:
```bash
# MusePose repo includes sample images in assets/ or examples/
ls MusePose/assets/
# pick a full-body standing person photo
```

Requirements for reference image:
- Full body visible (head to at least knees)
- Front-facing, neutral stance
- Clean background
- Portrait orientation preferred

### Test on 3 signs first
```bash
conda activate musepose
python step02_musepose_inference.py \
    --poses_dir    /path/to/dwpose_output \
    --reference    /path/to/reference_avatar.jpg \
    --output_dir   ./sign_videos_output \
    --musepose_dir /path/to/MusePose \
    --signs        maktab kitob qidirish
```

### Generate the rest/pause clip
```bash
# copy a neutral pose frame (hands at sides) as the rest pose source:
# use frame_000 from any sign where hands start near neutral
cp /path/to/dwpose_output/maktab/overlay/frame_000.png \
   /path/to/rest_pose/source.png
```

### Run all 50 signs
```bash
python step02_musepose_inference.py \
    --poses_dir    /path/to/dwpose_output \
    --reference    /path/to/reference_avatar.jpg \
    --output_dir   ./sign_videos_output \
    --musepose_dir /path/to/MusePose \
    --skip_existing   # safe to re-run, skips already-done signs
```

Estimated time on RTX 5090: ~15-30 minutes for all 50 signs.

### Verify outputs
```bash
python step03_verify_videos.py --videos_dir ./sign_videos_output
```

### Transfer back to Mac / web app
```bash
scp -r ./sign_videos_output/*.mp4 user@mac:/path/to/web_app/sign_videos/
```

---

## Expected output structure

```
web_app/
└── sign_videos/
    ├── rest.mp4                ← 8-frame neutral pause clip
    ├── assalomu_alaykum.mp4
    ├── bahor.mp4
    ├── birga.mp4
    └── ... (50 sign MP4s total)
```

---

## Troubleshooting

**DWPose gives blank/empty overlays:**
The signer may be cropped or too small. Check source video dimensions match
`--width 768 --height 432`. Try without resizing first.

**MusePose CUDA out of memory:**
RTX 5090 has 32GB — unlikely. If it happens, reduce `--width` and `--height`.

**MusePose script not found:**
Different MusePose versions have different entry points. Check the repo for
`inference.py`, `pose2vid.py`, or `test_stage2.py` and update step02 accordingly.

**Hand detail is blurry in output:**
This is a known MusePose limitation. If unacceptable for your use case,
the fallback is skeleton animation directly from `.npy` keypoints (Plan B).

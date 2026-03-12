"""
step01_extract_dwpose.py

Extracts DWPose keypoints from signer10 rep-0 videos.
Skips YOLOX detection — feeds full frame to pose model directly.
This works because your videos have consistent framing and background.

INSTALL:
  pip install opencv-python numpy tqdm onnxruntime huggingface_hub

USAGE:
  python step01_extract_dwpose.py \
      --dataset_root ../Data_Numpy_Arrays_RSL_UzSL \
      --output_dir   ./dwpose_output \
      --signs maktab kitob qidirish

  # all 50:
  python step01_extract_dwpose.py \
      --dataset_root ../Data_Numpy_Arrays_RSL_UzSL \
      --output_dir   ./dwpose_output
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# weights

def download_weights(weights_dir: Path):
    weights_dir.mkdir(parents=True, exist_ok=True)
    pose_path = weights_dir / "dw-ll_ucoco_384.onnx"
    if pose_path.exists():
        print(f"[setup] weights already at {weights_dir}")
        return pose_path
    print("[setup] downloading DWPose pose model (~134MB)...")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: pip install huggingface_hub"); sys.exit(1)
    hf_hub_download("yzd-v/DWPose", "dw-ll_ucoco_384.onnx", local_dir=str(weights_dir))
    print("[setup] done")
    return pose_path


# pose model

class PoseEstimator:
    """
    DW-LL-UCOCO pose model (SimCC).
    Input:  [1, 3, 384, 288]   (H=384, W=288, RGB, ImageNet-normalised)
    Output: simcc_x [1, 133, W*2]  simcc_y [1, 133, H*2]
    133 keypoints = 17 body + 6 feet + 68 face + 21 left hand + 21 right hand
    """

    POSE_H = 384
    POSE_W = 288
    SIMCC_SPLIT = 2  # bins = dimension * split

    # body connections (COCO 17-point)
    BODY_PAIRS = [
        (0,1),(0,2),(1,3),(2,4),        # head / neck
        (5,6),(5,7),(7,9),(6,8),(8,10), # shoulders / arms
        (5,11),(6,12),(11,12),          # torso to hips
        # legs removed — not visible in head-to-torso videos
    ]
    BODY_COLORS = [
        (255,0,0),(255,85,0),(255,170,0),(255,255,0),
        (170,255,0),(85,255,0),(0,255,0),(0,255,85),
        (0,255,170),(0,255,255),(0,170,255),(0,85,255),
        (0,0,255),(85,0,255),(170,0,255),(255,0,255),
    ]

    HAND_PAIRS = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
    ]

    def __init__(self, pose_onnx: Path):
        try:
            import onnxruntime as ort
        except ImportError:
            print("ERROR: pip install onnxruntime"); sys.exit(1)
        self.sess = ort.InferenceSession(
            str(pose_onnx), providers=["CPUExecutionProvider"]
        )

    def _preprocess(self, img_bgr: np.ndarray):
        """
        Letterbox-fit the image into [POSE_H, POSE_W] keeping aspect ratio.
        Returns (input_tensor, pad_top, pad_left, scale)
        so we can map keypoints back to the original image size.
        """
        oh, ow = img_bgr.shape[:2]
        scale  = min(self.POSE_W / ow, self.POSE_H / oh)
        nw     = int(ow * scale)
        nh     = int(oh * scale)

        resized  = cv2.resize(img_bgr, (nw, nh))
        pad_top  = (self.POSE_H - nh) // 2
        pad_left = (self.POSE_W - nw) // 2

        canvas = np.zeros((self.POSE_H, self.POSE_W, 3), dtype=np.float32)
        canvas[pad_top:pad_top+nh, pad_left:pad_left+nw] = resized.astype(np.float32)

        # BGR → RGB, ImageNet normalise
        canvas = canvas[:, :, ::-1].copy()
        mean = np.array([123.675, 116.28,  103.53 ], dtype=np.float32)
        std  = np.array([58.395,  57.12,   57.375 ], dtype=np.float32)
        canvas = (canvas - mean) / std

        inp = canvas.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
        return inp, pad_top, pad_left, scale, oh, ow

    def _decode_simcc(self, simcc_x: np.ndarray, simcc_y: np.ndarray,
                      pad_top: int, pad_left: int, scale: float,
                      orig_h: int, orig_w: int):
        """
        Decode SimCC logits → pixel coordinates in original image space.

        simcc_x shape: (133, POSE_W * SPLIT)  — one bin per sub-pixel along W
        simcc_y shape: (133, POSE_H * SPLIT)  — one bin per sub-pixel along H

        Each bin index maps to:   coord_in_pose_input = bin / SPLIT
        Then subtract padding and divide by scale to get original image coord.
        """
        # argmax gives the most likely sub-pixel position
        x_bins = np.argmax(simcc_x, axis=-1).astype(np.float32)  # (133,)
        y_bins = np.argmax(simcc_y, axis=-1).astype(np.float32)  # (133,)

        # convert bin index → position in pose input image (pixels)
        x_in_pose = x_bins / self.SIMCC_SPLIT
        y_in_pose = y_bins / self.SIMCC_SPLIT

        # remove letterbox padding and undo scale
        x_orig = (x_in_pose - pad_left) / scale
        y_orig = (y_in_pose - pad_top)  / scale

        # confidence: use the logit value at argmax, apply sigmoid
        num_kpts = simcc_x.shape[0]
        conf_x   = simcc_x[np.arange(num_kpts), np.argmax(simcc_x, axis=-1)]
        conf_y   = simcc_y[np.arange(num_kpts), np.argmax(simcc_y, axis=-1)]
        conf     = (1.0 / (1.0 + np.exp(-0.05 * (conf_x + conf_y))))

        # clip to image bounds
        x_orig = np.clip(x_orig, 0, orig_w - 1)
        y_orig = np.clip(y_orig, 0, orig_h - 1)

        return np.stack([x_orig, y_orig, conf], axis=-1)  # (133, 3)

    def infer(self, img_bgr: np.ndarray):
        """
        Run pose estimation on one frame.
        Returns keypoints: (133, 3) array  [x, y, confidence]
        """
        inp, pad_top, pad_left, scale, oh, ow = self._preprocess(img_bgr)
        out     = self.sess.run(None, {"input": inp})
        simcc_x = out[0][0]   # (133, W*2)
        simcc_y = out[1][0]   # (133, H*2)
        return self._decode_simcc(simcc_x, simcc_y, pad_top, pad_left, scale, oh, ow)

    def draw(self, canvas: np.ndarray, kpts: np.ndarray, conf_thr: float = 0.4):
        """Draw body + hand skeleton onto canvas (in-place). Returns canvas."""
        h, w = canvas.shape[:2]

        def pt(k):
            return (int(np.clip(kpts[k, 0], 0, w-1)),
                    int(np.clip(kpts[k, 1], 0, h-1)))

        def visible(k):
            return kpts[k, 2] > conf_thr

        # body (indices 0-16)
        for i, (a, b) in enumerate(self.BODY_PAIRS):
            if visible(a) and visible(b):
                cv2.line(canvas, pt(a), pt(b),
                         self.BODY_COLORS[i % len(self.BODY_COLORS)], 3, cv2.LINE_AA)
        for k in range(13):  # 0-12 only — skip knees/ankles (13-16)
            if visible(k):
                cv2.circle(canvas, pt(k), 4, (255, 255, 255), -1, cv2.LINE_AA)

        # left hand (indices 91-111, cyan)
        if len(kpts) > 111:
            lh = kpts[91:112]
            for a, b in self.HAND_PAIRS:
                if lh[a, 2] > conf_thr and lh[b, 2] > conf_thr:
                    cv2.line(canvas,
                             (int(lh[a,0]), int(lh[a,1])),
                             (int(lh[b,0]), int(lh[b,1])),
                             (0, 220, 255), 2, cv2.LINE_AA)
            for k in range(21):
                if lh[k, 2] > conf_thr:
                    cv2.circle(canvas, (int(lh[k,0]), int(lh[k,1])),
                               3, (0,220,255), -1, cv2.LINE_AA)

        # right hand (indices 112-132, orange)
        if len(kpts) > 132:
            rh = kpts[112:133]
            for a, b in self.HAND_PAIRS:
                if rh[a, 2] > conf_thr and rh[b, 2] > conf_thr:
                    cv2.line(canvas,
                             (int(rh[a,0]), int(rh[a,1])),
                             (int(rh[b,0]), int(rh[b,1])),
                             (255, 140, 0), 2, cv2.LINE_AA)
            for k in range(21):
                if rh[k, 2] > conf_thr:
                    cv2.circle(canvas, (int(rh[k,0]), int(rh[k,1])),
                               3, (255,140,0), -1, cv2.LINE_AA)

        return canvas


# video

def extract_frames(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def process_sign(estimator: PoseEstimator, video_path: Path,
                 out_dir: Path, target_w: int, target_h: int):
    overlay_dir = out_dir / "overlay"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    frames = extract_frames(video_path)
    if not frames:
        raise RuntimeError("No frames")

    all_poses = []
    for i, frame in enumerate(frames):
        frame = cv2.resize(frame, (target_w, target_h))

        kpts   = estimator.infer(frame)
        canvas = np.zeros_like(frame)
        estimator.draw(canvas, kpts)

        cv2.imwrite(str(overlay_dir / f"frame_{i:03d}.png"), canvas)
        all_poses.append(kpts.tolist())

    with open(out_dir / "poses.json", "w") as f:
        json.dump({"sign": out_dir.name, "num_frames": len(frames),
                   "width": target_w, "height": target_h,
                   "frames": all_poses}, f)
    return len(frames)


# main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--output_dir",   required=True)
    parser.add_argument("--signer",       default="signer10")
    parser.add_argument("--rep",          default="rep-0")
    parser.add_argument("--width",        type=int, default=768)
    parser.add_argument("--height",       type=int, default=432)
    parser.add_argument("--signs",        nargs="*", default=None)
    parser.add_argument("--weights_dir",  default="./dwpose_weights")
    parser.add_argument("--conf_thr",     type=float, default=0.4,
                        help="Keypoint confidence threshold for drawing (default 0.4)")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir   = Path(args.output_dir)
    signer_dir   = dataset_root / args.signer

    if not signer_dir.exists():
        print(f"ERROR: {signer_dir} not found"); sys.exit(1)

    all_signs = sorted([d.name for d in signer_dir.iterdir() if d.is_dir()])
    signs_to_process = [s for s in (args.signs or all_signs) if s in all_signs]
    missing = [s for s in (args.signs or []) if s not in all_signs]
    if missing:
        print(f"WARNING: not in dataset: {missing}")

    print(f"[info] signer:   {args.signer}")
    print(f"[info] rep:      {args.rep}")
    print(f"[info] signs:    {len(signs_to_process)} to process")
    print(f"[info] output:   {output_dir}")
    print(f"[info] size:     {args.width}×{args.height}")
    print(f"[info] no YOLOX — full-frame pose (consistent framing assumed)\n")

    pose_onnx = download_weights(Path(args.weights_dir))
    print("[setup] loading pose model...")
    estimator = PoseEstimator(pose_onnx)
    print("[setup] ready\n")

    failed = []
    for sign_name in tqdm(signs_to_process, desc="Extracting poses"):
        video_path = signer_dir / sign_name / "videos" / args.rep / "video.mp4"
        if not video_path.exists():
            tqdm.write(f"  SKIP {sign_name}: not found")
            failed.append(sign_name); continue

        sign_out = output_dir / sign_name
        sign_out.mkdir(parents=True, exist_ok=True)
        try:
            n = process_sign(estimator, video_path, sign_out, args.width, args.height)
            tqdm.write(f"  ✓ {sign_name} ({n} frames)")
        except Exception as e:
            tqdm.write(f"  ✗ {sign_name}: {e}")
            failed.append(sign_name)

    print(f"\nDone. {len(signs_to_process)-len(failed)}/{len(signs_to_process)} extracted.")
    if failed:
        print(f"Failed: {failed}")
    print(f"\nCheck: open {output_dir}/maktab/overlay/frame_010.png")
    print("Body = coloured lines, left hand = cyan, right hand = orange")
    print("If hands are missing, lower --conf_thr to 0.2 and rerun.")

if __name__ == "__main__":
    main()
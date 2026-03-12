"""
step02_musepose_inference.py
Runs MusePose inference for all 50 signs.
Must be run from inside the MusePose/ directory.

USAGE (from inside MusePose/):
  # test 3 signs first:
  python step02_musepose_inference.py \
      --poses_dir    ../../dwpose_output \
      --reference    /path/to/avatar.jpg \
      --output_dir   ../../sign_videos \
      --signs        maktab kitob qidirish

  # all 50 signs:
  python step02_musepose_inference.py \
      --poses_dir    ../../dwpose_output \
      --reference    /path/to/avatar.jpg \
      --output_dir   ../../sign_videos
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def frames_to_video(frames_dir: Path, output_path: Path, fps: int = 30):
    """Convert overlay/frame_000.png ... frame_031.png into an MP4."""
    frames = sorted(frames_dir.glob("frame_*.png"))
    if not frames:
        raise RuntimeError(f"No frame_*.png in {frames_dir}")

    # Use glob input to avoid apostrophe issues with concat list files
    # frame pattern: /path/to/overlay/frame_%03d.png
    pattern = str(frames_dir / "frame_%03d.png")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", pattern,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-vf", f"fps={fps}",
        str(output_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")


def run_pose_align(reference: Path, pose_video: Path,
                   aligned_video: Path, align_viz: Path,
                   detect_res: int, image_res: int):
    """
    pose_align.py arguments (from --help):
      --imgfn_refer              reference image path
      --vidfn                    input pose video path
      --outfn_align_pose_video   output aligned pose video
      --outfn                    output alignment visualisation
      --detect_resolution        detection resolution
      --image_resolution         image resolution
      --align_frame              frame index to align on (use 0)
      --max_frame                max frames to process
    """
    aligned_video.parent.mkdir(parents=True, exist_ok=True)
    align_viz.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "pose_align.py",
        "--imgfn_refer",             str(reference),
        "--vidfn",                   str(pose_video),
        "--outfn_align_pose_video",  str(aligned_video),
        "--outfn",                   str(align_viz),
        "--detect_resolution",       str(detect_res),
        "--image_resolution",        str(image_res),
        "--align_frame",             "0",
        "--max_frame",               "32",
    ]

    # hide GPU — pose_align uses old PyTorch incompatible with RTX 5090 (sm_120)
    env_cpu = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run(cmd, capture_output=True, text=True, env=env_cpu)
    if result.returncode != 0:
        raise RuntimeError(
            f"pose_align.py failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def update_test_config(reference: Path, aligned_video: Path):
    """Point test_stage_2.yaml test_cases at our reference + aligned pose."""
    import yaml
    config_path = Path("configs/test_stage_2.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # use absolute paths to avoid any relative-path confusion
    cfg["test_cases"] = {
        str(reference): [str(aligned_video)]
    }

    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    print(f"  [config] reference: {reference.name}")
    print(f"  [config] pose:      {aligned_video.name}")


def run_test_stage_2(width: int = 768, height: int = 432, num_frames: int = 32):
    """Run MusePose test_stage_2.py and return path to generated MP4."""
    # ensure GPU is visible (in case parent process had CUDA_VISIBLE_DEVICES="")
    env_gpu = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    # slice must be <= num_frames; overlap must be < slice
    slice_frames   = min(num_frames, 24)
    overlap_frames = min(4, slice_frames - 1)
    result = subprocess.run(
        [
            sys.executable, "test_stage_2.py",
            "-W", str(width),
            "-H", str(height),
            "-L", str(num_frames),
            "-S", str(slice_frames),
            "-O", str(overlap_frames),
        ],
        capture_output=True, text=True, env=env_gpu
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"test_stage_2.py failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    # find newest MP4 in output/
    output_dir = Path("output")
    if not output_dir.exists():
        raise RuntimeError("test_stage_2.py ran but no output/ directory found")

    mp4s = sorted(output_dir.glob("**/*.mp4"), key=lambda p: p.stat().st_mtime)
    if not mp4s:
        raise RuntimeError("test_stage_2.py ran but no MP4 found in output/")

    return mp4s[-1]


def process_one_sign(sign_name: str, poses_dir: Path, reference: Path,
                     output_dir: Path, tmp_dir: Path,
                     width: int, height: int):

    overlay_dir = poses_dir / sign_name / "overlay"
    if not overlay_dir.exists():
        raise RuntimeError(f"overlay dir not found: {overlay_dir}")

    # A: frames → pose video
    print(f"  [A] converting frames to video...")
    pose_video = tmp_dir / f"{sign_name}_pose.mp4"
    frames_to_video(overlay_dir, pose_video)

    # B: skip pose_align — all signers share same framing, feed pose directly
    # pose_align re-runs DWPose on the skeleton overlay which produces garbage
    print(f"  [B] skipping pose_align (consistent framing, not needed)...")

    # C: update config + run MusePose  
    print(f"  [C] running MusePose inference...")
    update_test_config(reference, pose_video)
    musepose_out = run_test_stage_2(width=width, height=height, num_frames=32)

    # D: copy to output
    final_path = output_dir / f"{sign_name}.mp4"
    shutil.copy(musepose_out, final_path)
    return final_path


def main():
    parser = argparse.ArgumentParser(
        description="Run MusePose for all 50 signs. Run from inside MusePose/ directory."
    )
    parser.add_argument("--poses_dir",    required=True,
                        help="Path to dwpose_output/ (from step01)")
    parser.add_argument("--reference",    required=True,
                        help="Path to avatar reference image (.jpg/.png)")
    parser.add_argument("--output_dir",   required=True,
                        help="Where to save final MP4s")
    parser.add_argument("--width",        type=int, default=768,
                        help="Output width (default: 768)")
    parser.add_argument("--height",       type=int, default=432,
                        help="Output height (default: 432)")
    parser.add_argument("--signs",        nargs="*", default=None,
                        help="Only process these signs (default: all)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip signs that already have an output MP4")
    args = parser.parse_args()

    # guard: must be run from MusePose/
    if not Path("test_stage_2.py").exists():
        print("ERROR: run this script from inside the MusePose/ directory:")
        print("  cd .../musepose/MusePose")
        print("  python step02_musepose_inference.py ...")
        sys.exit(1)

    poses_dir  = Path(args.poses_dir).resolve()
    reference  = Path(args.reference).resolve()
    output_dir = Path(args.output_dir).resolve()
    tmp_dir    = Path("_tmp_sign_production").resolve()

    for p, name in [(reference, "reference"), (poses_dir, "poses_dir")]:
        if not p.exists():
            print(f"ERROR: {name} not found: {p}"); sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # ensure pyyaml available
    try:
        import yaml
    except ImportError:
        os.system(f"{sys.executable} -m pip install pyyaml")
        import yaml

    # discover available signs
    all_signs = sorted([
        d.name for d in poses_dir.iterdir()
        if d.is_dir() and (d / "overlay").exists()
    ])
    signs = [s for s in (args.signs or all_signs) if s in all_signs]
    bad   = [s for s in (args.signs or []) if s not in all_signs]
    if bad:
        print(f"WARNING: signs not found in poses_dir: {bad}")

    print(f"\n{'='*60}")
    print(f"  MusePose Sign Production")
    print(f"{'='*60}")
    print(f"  reference : {reference}")
    print(f"  poses_dir : {poses_dir}")
    print(f"  output    : {output_dir}")
    print(f"  resolution: {args.width}×{args.height}")
    print(f"  signs     : {len(signs)} to process")
    print(f"{'='*60}\n")

    failed = []
    for i, sign_name in enumerate(signs):
        final_path = output_dir / f"{sign_name}.mp4"

        if args.skip_existing and final_path.exists():
            print(f"[{i+1}/{len(signs)}] SKIP {sign_name} (already exists)")
            continue

        print(f"[{i+1}/{len(signs)}] {sign_name}")
        try:
            result = process_one_sign(
                sign_name, poses_dir, reference,
                output_dir, tmp_dir, args.width, args.height
            )
            mb = result.stat().st_size / 1024 / 1024
            print(f"  ✓ {result.name} ({mb:.1f} MB)\n")
        except Exception as e:
            print(f"  ✗ FAILED: {e}\n")
            failed.append(sign_name)

    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\n{'='*60}")
    print(f"  Done: {len(signs)-len(failed)}/{len(signs)} generated")
    if failed:
        print(f"  Failed: {failed}")
        print(f"  Retry:  --signs {' '.join(failed)} --skip_existing")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")
    print("\nNext: copy sign_videos/*.mp4 to web_app/sign_videos/ on your Mac")


if __name__ == "__main__":
    main()
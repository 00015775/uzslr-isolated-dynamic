"""
step03_verify_videos.py

Verifies all 50 generated sign videos before copying to the web app.
Checks: file exists, is readable, correct fps, minimum frame count.
Run this on Ubuntu after step02 completes.

USAGE:
  python step03_verify_videos.py --videos_dir /path/to/sign_videos_output
"""

import argparse
import json
from pathlib import Path

import cv2

# the canonical 50 sign names — update if yours differ
EXPECTED_SIGNS = [
    "assalomu_alaykum", "bahor", "birga", "bosh_kiyim", "boshlanishi",
    "bo'sh", "bozor", "eshik", "futbol", "iltimos",
    "internet", "javob", "jismoniy_tarbiya", "karam", "kartoshka",
    "kichik", "kitob", "ko'prik", "likopcha", "maktab",
    "mehmonxona", "mehribon", "metro", "musiqa", "o'simlik_yog'i",
    "o'ynash", "ochish", "ot", "ovqat_tayyorlash", "oxiri",
    "poezd", "pomidor", "qidirish", "qish", "qo'ziqorin",
    "qor", "qorong'i", "quyon", "restoran", "sariyog'",
    "shokolad", "sovun", "stakan", "televizor", "tosh",
    "toza", "turish", "yomg'ir", "yopish", "yordam_berish",
]


def check_video(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"ok": False, "error": "cannot open"}

    fps        = cap.get(cv2.CAP_PROP_FPS)
    framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    issues = []
    if framecount < 20:
        issues.append(f"too few frames ({framecount})")
    if fps < 10:
        issues.append(f"low fps ({fps:.1f})")
    if width == 0 or height == 0:
        issues.append("zero dimensions")

    size_mb = path.stat().st_size / 1024 / 1024
    return {
        "ok":     len(issues) == 0,
        "frames": framecount,
        "fps":    round(fps, 1),
        "size":   f"{width}×{height}",
        "mb":     round(size_mb, 2),
        "issues": issues,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos_dir", required=True,
                        help="Directory containing sign MP4s")
    parser.add_argument("--save_report", default="verification_report.json",
                        help="Save JSON report to this file")
    args = parser.parse_args()

    videos_dir = Path(args.videos_dir)
    if not videos_dir.exists():
        print(f"ERROR: {videos_dir} does not exist")
        return

    print(f"Checking videos in: {videos_dir}\n")

    report   = {}
    missing  = []
    failed   = []
    ok_count = 0

    for sign in EXPECTED_SIGNS:
        path = videos_dir / f"{sign}.mp4"
        if not path.exists():
            print(f"  MISSING  {sign}.mp4")
            missing.append(sign)
            report[sign] = {"ok": False, "error": "missing"}
            continue

        result = check_video(path)
        report[sign] = result

        if result["ok"]:
            print(f"  ✓  {sign:<30} {result['frames']}f  {result['fps']}fps  "
                  f"{result['size']}  {result['mb']}MB")
            ok_count += 1
        else:
            issues = ", ".join(result["issues"])
            print(f"  ✗  {sign:<30} ISSUES: {issues}")
            failed.append(sign)

    # check rest clip
    rest_path = videos_dir / "rest.mp4"
    if rest_path.exists():
        r = check_video(rest_path)
        status = "✓" if r["ok"] else "✗"
        print(f"\n  {status}  rest.mp4  (inter-sign pause clip)")
    else:
        print(f"\n  MISSING  rest.mp4  — generate with step02 --signs rest")

    print(f"\n{'─'*60}")
    print(f"Total:   {len(EXPECTED_SIGNS)} expected")
    print(f"OK:      {ok_count}")
    print(f"Missing: {len(missing)}  {missing if missing else ''}")
    print(f"Failed:  {len(failed)}   {failed if failed else ''}")

    with open(args.save_report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {args.save_report}")

    if missing or failed:
        print("\nRe-run step02 for missing/failed signs:")
        retry = missing + failed
        print(f"  --signs {' '.join(retry)}")
    else:
        print("\nAll videos OK. Copy to web_app/sign_videos/ and proceed to web integration.")


if __name__ == "__main__":
    main()

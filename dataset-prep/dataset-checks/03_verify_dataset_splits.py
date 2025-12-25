"""
(READ-ONLY file)

This script inspects the train/validation/test dataset splits and reports:
- Total number of repetition folders in each split
- Whether all expected signs are present in each split
- Per-sign repetition counts for every split
"""

import os

# Post-Splitting dataset location at the root level
DATA_ROOT = "../../data"

SPLITS = ["train", "validation", "test"]

# 50 default signs
DEFAULT_SIGNS = ['assalomu_alaykum', 'bahor', 'birga', "bo'sh", 'bosh_kiyim', 'boshlanishi', 'bozor', 'eshik', 
               'futbol', 'iltimos', 'internet', 'javob', 'jismoniy_tarbiya', 'karam', 'kartoshka', 
               'kichik', 'kitob', "ko'prik", 'likopcha', 'maktab', 'mehmonxona', 'mehribon', 'metro', 
               'musiqa', "o'simlik_yog'i", "o'ynash", 'ochish', 'ot', 'ovqat_tayyorlash', 
               'oxiri', 'poezd', 'pomidor', 'qidirish', 'qish', "qo'ziqorin", 'qor', "qorong'i", 'quyon', 
               'restoran', "sariyog'", 'shokolad', 'sovun', 'stakan', 'televizor', 'tosh', 'toza',
               'turish', "yomg'ir", 'yopish', 'yordam_berish']


print("\n=== DATASET SPLIT VERIFICATION (READ-ONLY) ===\n")

total_reps_per_split = {}
missing_signs = {split: [] for split in SPLITS}

for split in SPLITS:
    split_path = os.path.join(DATA_ROOT, split)
    total_reps = 0

    print(f"\n--- {split.upper()} ---")

    for sign in DEFAULT_SIGNS:
        sign_path = os.path.join(split_path, sign)

        if not os.path.isdir(sign_path):
            missing_signs[split].append(sign)
            print(f"  [MISSING] {sign}")
            continue

        reps = [
            d for d in os.listdir(sign_path)
            if os.path.isdir(os.path.join(sign_path, d)) and d.startswith("rep-")
        ]

        rep_count = len(reps)
        total_reps += rep_count

        print(f"  {sign:<25} -> {rep_count} reps")

    total_reps_per_split[split] = total_reps
    print(f"\nTotal repetitions in {split}: {total_reps}")

print("\n=== SUMMARY ===")

for split in SPLITS:
    print(f"\n{split.upper()}:")
    print(f"  Total repetitions: {total_reps_per_split[split]}")
    print(f"  Missing signs: {len(missing_signs[split])}")

    if missing_signs[split]:
        for sign in missing_signs[split]:
            print(f"    - {sign}")

print("\nVerification completed. No files were modified.")

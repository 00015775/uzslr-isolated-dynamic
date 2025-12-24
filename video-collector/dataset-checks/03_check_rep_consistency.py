"""
(READ-ONLY file)
Data consistency checker for each repetitions present in landmarks and videos folder.

This script performs two main validation checks on the processed dataset structure:
1. For each signer and sign, verifies that the repetition folders (rep_XX) are identical 
   between 'landmarks' and 'videos' directories.
2. Ensures that every signer has exactly the expected set of 50 cleaned signs 
   as defined in CLEAN_SIGNS, reporting any missing or extra sign folders.

Helps detect missing data, synchronization issues between landmarks and videos folders, or outdated 
sign lists after cleaning/pruning.
"""


import os

# Base directory containing signer01 to signer10
BASE_DIR = "../Data_Numpy_Arrays_RSL_UzSL"

def get_rep_folders(path):
    if not os.path.isdir(path):
        return set()
    return {d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))}

def get_sign_folders(signer_path):
    return {
        d for d in os.listdir(signer_path)
        if os.path.isdir(os.path.join(signer_path, d))
    }

print("\n=== REP FOLDER CONSISTENCY CHECK ===\n")

signer_sign_sets = {}
all_signers = sorted(
    d for d in os.listdir(BASE_DIR)
    if os.path.isdir(os.path.join(BASE_DIR, d)) and d.startswith("signer")
)

for signer in all_signers:
    signer_path = os.path.join(BASE_DIR, signer)
    sign_folders = get_sign_folders(signer_path)
    signer_sign_sets[signer] = sign_folders

    print(f"\nSigner: {signer}")
    print("-" * (8 + len(signer)))

    for sign in sorted(sign_folders):
        sign_path = os.path.join(signer_path, sign)
        landmarks_path = os.path.join(sign_path, "landmarks")
        videos_path = os.path.join(sign_path, "videos")

        landmarks_reps = get_rep_folders(landmarks_path)
        videos_reps = get_rep_folders(videos_path)

        if landmarks_reps == videos_reps:
            print(f"  ✔ {sign}: all rep folders match ({sorted(landmarks_reps)})")
        else:
            print(f"  ✖ {sign}: mismatch detected")
            print(f"     landmarks reps: {sorted(landmarks_reps)}")
            print(f"     videos reps:    {sorted(videos_reps)}")

            missing_in_videos = landmarks_reps - videos_reps
            extra_in_videos = videos_reps - landmarks_reps

            if missing_in_videos:
                print(f"     missing in videos: {sorted(missing_in_videos)}")
            if extra_in_videos:
                print(f"     extra in videos:   {sorted(extra_in_videos)}")


print("\n=== SIGN FOLDER CHECK AGAINST CLEAN_SIGNS ===\n")

# After data cleaning and prunning, 50 signs were left. CLEAN_SIGNS
# MAKE SURE TO KEEP THIS DICTIONARY UPDATED IN CASE IF SIGNS ARE ADDED OR REMOVED
CLEAN_SIGNS = ['assalomu_alaykum', 'bahor', 'birga', "bo'sh", 'bosh_kiyim', 'boshlanishi', 'bozor', 'eshik', 
               'futbol', 'iltimos', 'internet', 'javob', 'jismoniy_tarbiya', 'karam', 'kartoshka', 
               'kichik', 'kitob', "ko'prik", 'likopcha', 'maktab', 'mehmonxona', 'mehribon', 'metro', 
               'musiqa', "o'simlik_yog'i", "o'ynash", 'ochish', 'ot', 'ovqat_tayyorlash', 
               'oxiri', 'poezd', 'pomidor', 'qidirish', 'qish', "qo'ziqorin", 'qor', "qorong'i", 'quyon', 
               'restoran', "sariyog'", 'shokolad', 'sovun', 'stakan', 'televizor', 'tosh', 'toza',
               'turish', "yomg'ir", 'yopish', 'yordam_berish']

CLEAN_SIGNS_SET = set(CLEAN_SIGNS)


for signer, signer_signs in signer_sign_sets.items():
    signer_signs_set = set(signer_signs)

    missing = CLEAN_SIGNS_SET - signer_signs_set
    extra = signer_signs_set - CLEAN_SIGNS_SET

    if not missing and not extra:
        print(f"✔ {signer}: all {len(CLEAN_SIGNS_SET)} signs match CLEAN_SIGNS")
    else:
        print(f"✖ {signer}: sign mismatch detected")

        if missing:
            print(f"  missing signs ({len(missing)}):")
            for s in sorted(missing):
                print(f"    {s}")

        if extra:
            print(f"  extra signs ({len(extra)}):")
            for s in sorted(extra):
                print(f"    {s}")

        print()

print("\n=== CHECK COMPLETE ===\n")


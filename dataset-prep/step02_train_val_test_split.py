"""
(WRITE-OUT file) USE WITH CAUTION!

This script creates a stratified train/validation/test split of the dataset while keeping the original data completely untouched.

Key features:
- Copies (does NOT move or modify) repetition folders (rep-XX) from the original structure.
- Splits at the repetition level for each sign independently.
- Uses 80% train / 10% validation / 10% test ratio per sign.
- Ensures every sign appears in all three splits (as long as it has enough repetitions).
- Fully reproducible: fixed random seed (42) guarantees the same split every run.
- Safe: skips copying if a folder already exists, never overwrites anything.
- Original data under data/{sign}/ remains 100% unchanged.

After running, there will be train/validation/test subfolders within the existing "data" folder at the root level:
    data/train/{sign}/rep-XX/...
    data/validation/{sign}/rep-XX/...
    data/test/{sign}/rep-XX/...

NOTES:
This code does not create a new "data" folder at the root level, but rather it creates train/validation/test subfolders within the existing
"data" folder (which was created with the step01_reorganize_dataset.py). It is advised to manually remove {sign-name} 
subfolders from the "data" folder since landmarks are basically copied into train/val/test splits, effectively the storage doubles.
"""

import os
import shutil
import random

# Fixed random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# 50 default signs
DEFAULT_SIGNS = ['assalomu_alaykum', 'bahor', 'birga', "bo'sh", 'bosh_kiyim', 'boshlanishi', 'bozor', 'eshik', 
               'futbol', 'iltimos', 'internet', 'javob', 'jismoniy_tarbiya', 'karam', 'kartoshka', 
               'kichik', 'kitob', "ko'prik", 'likopcha', 'maktab', 'mehmonxona', 'mehribon', 'metro', 
               'musiqa', "o'simlik_yog'i", "o'ynash", 'ochish', 'ot', 'ovqat_tayyorlash', 
               'oxiri', 'poezd', 'pomidor', 'qidirish', 'qish', "qo'ziqorin", 'qor', "qorong'i", 'quyon', 
               'restoran', "sariyog'", 'shokolad', 'sovun', 'stakan', 'televizor', 'tosh', 'toza',
               'turish', "yomg'ir", 'yopish', 'yordam_berish']


# Pre-Splitting dataset location at the root level
data_root = '../data'

# Desired splits
splits = ['train', 'validation', 'test']

# Create split directories if they do not exist
for split in splits:
    os.makedirs(os.path.join(data_root, split), exist_ok=True)

# Process each sign independently
for sign in DEFAULT_SIGNS:
    sign_path = os.path.join(data_root, sign)
    
    if not os.path.exists(sign_path):
        print(f"Warning: Sign folder not found: {sign_path}")
        continue
    
    # Get list of repetition folders (rep-XX)
    reps = [d for d in os.listdir(sign_path) if os.path.isdir(os.path.join(sign_path, d)) and d.startswith('rep-')]
    
    if len(reps) == 0:
        print(f"Warning: No repetitions found for sign: {sign}")
        continue
    
    # Shuffle the repetitions with the fixed seed
    random.shuffle(reps)
    
    # Calculate split sizes
    total_reps = len(reps)
    train_count = int(0.8 * total_reps)
    val_count = int(0.1 * total_reps)
    test_count = total_reps - train_count - val_count  # Use remainder for test to avoid rounding issues
    
    # Assign reps to splits
    train_reps = reps[:train_count]
    val_reps = reps[train_count:train_count + val_count]
    test_reps = reps[train_count + val_count:]
    
    print(f"{sign}: {total_reps} reps -> train:{len(train_reps)}, val:{len(val_reps)}, test:{len(test_reps)}")
    
    # Copy reps to each split
    for split, split_reps in zip(splits, [train_reps, val_reps, test_reps]):
        if len(split_reps) == 0:
            print(f"  Warning: No repetitions assigned to {split} for sign '{sign}'")
            continue
            
        split_sign_path = os.path.join(data_root, split, sign)
        os.makedirs(split_sign_path, exist_ok=True)
        
        for rep in split_reps:
            src_rep_path = os.path.join(sign_path, rep)
            dest_rep_path = os.path.join(split_sign_path, rep)
            
            # Copy only if destination does not exist (safe, no overwrite)
            if os.path.exists(dest_rep_path):
                print(f"  Skipping (already exists): {dest_rep_path}")
            else:
                shutil.copytree(src_rep_path, dest_rep_path)
                print(f"  Copied: {src_rep_path} → {dest_rep_path}")

print("\nDataset splitting completed successfully!")
print("Original data remains untouched. New structure created under data/train, data/validation, data/test")
print(f"Random seed used: {RANDOM_SEED} (for full reproducibility)")



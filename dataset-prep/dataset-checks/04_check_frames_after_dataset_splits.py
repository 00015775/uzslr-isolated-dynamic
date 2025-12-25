"""
(READ-ONLY file)

This script scans all dataset splits (train, validation, test), signs, and repetitions,
and counts the number of frame-XX.npy files in each rep-XX folder.
It prints the counts for all folders and flags any rep-XX that do not contain exactly 32 files.
"""

import os
from pathlib import Path

# Post-Splitting dataset location at the root level
DATA_DIR = Path('../../data')

SPLITS = ['train', 'validation', 'test']

def check_frame_counts():
    print("Checking frame counts in each repetition folder across all splits...\n")
    
    all_good = True
    
    for split in SPLITS:
        split_dir = DATA_DIR / split
        if not split_dir.exists():
            print(f"Warning: Split folder not found: {split_dir}")
            continue
        
        print(f"\nSplit: {split}")
        
        for sign_dir in sorted(split_dir.iterdir()):
            if not sign_dir.is_dir():
                continue
            
            print(f" Sign: {sign_dir.name}")
            
            for rep_dir in sorted(sign_dir.iterdir()):
                if not rep_dir.is_dir() or not rep_dir.name.startswith('rep-'):
                    continue
                
                # Find all .npy files in this rep folder
                npy_files = list(rep_dir.glob('frame-*.npy'))
                count = len(npy_files)
                
                if count == 32:
                    print(f"  {rep_dir.name}: {count} frames -> OK")
                else:
                    print(f"  {rep_dir.name}: {count} frames -> WARNING (expected 32)")
                    all_good = False
    
    print("\n" + "="*50)
    if all_good:
        print("All repetition folders have exactly 32 frames. Perfect!")
    else:
        print("Some repetition folders do NOT have exactly 32 frames.")

if __name__ == "__main__":
    if not DATA_DIR.exists():
        print(f"Error: '{DATA_DIR}' folder not found.")
    else:
        check_frame_counts()


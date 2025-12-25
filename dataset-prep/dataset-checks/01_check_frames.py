"""
(READ-ONLY file)

Check that every repetition folder (rep-X) for each sign contains exactly 32 frame-XX.npy files.
Reports any folders with missing or extra frames.
"""

import os
from pathlib import Path

# Path to the newly reorganized data folder
DATA_DIR = Path('../../data')

def check_frame_counts():
    print("Checking frame counts in each repetition folder...\n")
    
    all_good = True
    
    for sign_dir in sorted(DATA_DIR.iterdir()):
        if not sign_dir.is_dir():
            continue
        
        print(f"Sign: {sign_dir.name}")
        
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


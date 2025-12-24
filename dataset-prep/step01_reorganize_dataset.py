"""
(WRITE-OUT file) USE WITH CAUTION!

Safely reorganizes the UzSL landmark dataset by copying all repetition folders 
from video-collector/Data_Numpy_Arrays_RSL_UzSL/signerXX/{sign}/landmarks/rep-XX/ 
into a new clean structure: data/{sign}/rep-1/, rep-2/, ...

- Copies only landmark data (original dataset remains untouched)
- Merges repetitions from all 10 signers for each of the 50 signs
- Renames folders sequentially (rep-1, rep-2, ...) to prevent conflicts
- Ignores 'videos/' folders entirely
- Prints progress and a summary of repetitions per sign
"""

import os
import shutil
from pathlib import Path

# 50 default signs
DEFAULT_SIGNS = ['assalomu_alaykum', 'bahor', 'birga', "bo'sh", 'bosh_kiyim', 'boshlanishi', 'bozor', 'eshik', 
               'futbol', 'iltimos', 'internet', 'javob', 'jismoniy_tarbiya', 'karam', 'kartoshka', 
               'kichik', 'kitob', "ko'prik", 'likopcha', 'maktab', 'mehmonxona', 'mehribon', 'metro', 
               'musiqa', "o'simlik_yog'i", "o'ynash", 'ochish', 'ot', 'ovqat_tayyorlash', 
               'oxiri', 'poezd', 'pomidor', 'qidirish', 'qish', "qo'ziqorin", 'qor', "qorong'i", 'quyon', 
               'restoran', "sariyog'", 'shokolad', 'sovun', 'stakan', 'televizor', 'tosh', 'toza',
               'turish', "yomg'ir", 'yopish', 'yordam_berish']


def copy_and_reorganize_dataset():
    # Original dataset location
    base_path = Path('../video-collector/Data_Numpy_Arrays_RSL_UzSL')
    
    # New dataset location at the root level
    target_base = Path('../data')
    target_base.mkdir(exist_ok=True)
    
    # Counter to assign unique rep numbers per sign (starting from 1)
    sign_rep_counter = {sign: 0 for sign in DEFAULT_SIGNS}
    
    total_copied = 0
    
    print("Starting safe copy and reorganization (original data will NOT be modified)...\n")
    
    # Loop through each signer (signer01 to signer10)
    for signer_dir in sorted(base_path.iterdir()):
        if not signer_dir.is_dir() or not signer_dir.name.startswith('signer'):
            continue
        
        print(f"Processing {signer_dir.name}...")
        
        for sign_dir in signer_dir.iterdir():
            if not sign_dir.is_dir() or sign_dir.name not in DEFAULT_SIGNS:
                continue
            
            sign_name = sign_dir.name
            landmarks_dir = sign_dir / 'landmarks'
            
            if not landmarks_dir.exists():
                print(f"  Warning: No landmarks folder found for {sign_name} in {signer_dir.name}")
                continue
            
            # Process each repetition folder inside landmarks/
            for rep_dir in sorted(landmarks_dir.iterdir()):
                if rep_dir.is_dir() and rep_dir.name.startswith('rep-'):
                    # Increment and get new sequential name
                    sign_rep_counter[sign_name] += 1
                    new_rep_num = sign_rep_counter[sign_name]
                    new_rep_name = f'rep-{new_rep_num}'
                    
                    # Create target directories
                    target_sign_dir = target_base / sign_name
                    target_sign_dir.mkdir(exist_ok=True)
                    target_rep_dir = target_sign_dir / new_rep_name
                    
                    # COPY the entire repetition folder, much safer than shutil.move()
                    shutil.copytree(str(rep_dir), str(target_rep_dir))
                    total_copied += 1
                    print(f"  Copied {rep_dir.relative_to(base_path.parent)} -> data/{sign_name}/{new_rep_name}/")
    
    print("\nCopy and reorganization completed safely!")
    print(f"Total repetition folders copied: {total_copied}")
    print(f"New dataset located at: {target_base.resolve()}")
    
    print("\nSummary per sign:")
    for sign in sorted(DEFAULT_SIGNS):
        count = sign_rep_counter[sign]
        if count > 0:
            print(f"  {sign}: {count} repetitions")
        else:
            print(f"  {sign}: 0 repetitions (warning: missing?)")

if __name__ == "__main__":
    copy_and_reorganize_dataset()


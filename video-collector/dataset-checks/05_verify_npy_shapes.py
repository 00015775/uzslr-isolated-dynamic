"""
(READ-ONLY file)
This file verifies that every frame-*.npy file in the dataset has the exact shape (1662,).

It is completely READ-ONLY: it only loads each .npy file to check its shape using np.load(),
with no writing, saving, or modification of any file.

The script walks through the full directory structure (signers -> signs -> landmarks -> repetitions -> frames),
counts all checked files, reports progress per signer, and lists any file with a wrong shape or loading error.
"""

import os
import numpy as np
from pathlib import Path

# Base directory containing signer01 to signer10
root_dir = Path('../Data_Numpy_Arrays_RSL_UzSL')

# Verify the directory exists
if not root_dir.exists():
    print(f"Error: The directory {root_dir} does not exist. Please check the path.")
    exit()

if not root_dir.is_dir():
    print(f"Error: {root_dir} is not a directory.")
    exit()

all_correct = True
issues = []
total_files = 0

print("Starting shape verification (read-only)...\n")

for signer_dir in sorted(root_dir.iterdir()):
    if not signer_dir.is_dir() or not signer_dir.name.startswith('signer'):
        continue
    
    print(f"Processing {signer_dir.name}...")
    
    for sign_dir in signer_dir.iterdir():
        if not sign_dir.is_dir():
            continue
        
        landmarks_dir = sign_dir / 'landmarks'
        if not landmarks_dir.exists() or not landmarks_dir.is_dir():
            continue
        
        for rep_dir in landmarks_dir.iterdir():
            if not rep_dir.is_dir() or not rep_dir.name.startswith('rep-'):
                continue
            
            for npy_path in rep_dir.glob('frame-*.npy'):
                total_files += 1
                try:
                    arr = np.load(npy_path)  # Loads the array without modifying the file
                    if arr.shape != (1662,):
                        all_correct = False
                        issues.append(f"Wrong shape: {npy_path.relative_to(root_dir)} -> {arr.shape}")
                except Exception as e:
                    all_correct = False
                    issues.append(f"Error loading: {npy_path.relative_to(root_dir)} -> {str(e)}")

print("\n" + "="*50)
print(f"Total frame-*.npy files checked: {total_files}")

if all_correct:
    print("\nSUCCESS: All .npy files have the correct shape (1662,)")
else:
    print("\nWARNING: Found issues with the following files:")
    for issue in issues:
        print(f"  - {issue}")

print("\nVerification complete. No files were modified.")


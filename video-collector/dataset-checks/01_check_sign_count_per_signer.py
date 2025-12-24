"""
(READ-ONLY file)
Verifies that each signer directory contains the expected number of sign folders.

This script scans signer01 through signer10 in the processed Uzbek Sign Language 
dataset and counts the number of sign subdirectories (excluding any hidden folders).
It then compares the count against the expected total 
(currently set to 50 signs after cleaning/pruning).

Output: A per-signer report showing the actual count and whether it matches the 
expected number.

Useful for quickly confirming dataset completeness after adding, removing, or 
pruning signs. Remember to update `total_signs_expected` whenever the sign list changes.
"""


import os
from pathlib import Path

# Base directory containing signer01 to signer10
base_dir = Path('../Data_Numpy_Arrays_RSL_UzSL')

# Signer folders: signer01 to signer10
signer_folders = [f"signer{i:02d}" for i in range(1, 11)]

# MAKE SURE TO CHANGE THIS NUMBER IN CASE NEW SIGNS ARE ADDED OR REMOVED
total_signs_expected = 50

for signer in signer_folders:
    signer_path = base_dir / signer
    if not signer_path.exists():
        print(f"{signer}: Directory not found")
        continue

    # Get all sign folders, excluding possible hidden files or meta files
    sign_folders = [item for item in signer_path.iterdir() 
                    if item.is_dir() and not item.name.startswith('.') and item.name != '__pycache__']

    count = len(sign_folders)
    status = "✓ Matches" if count == total_signs_expected else "✗ Does NOT match"
    
    print(f"{signer}: {count} sign folders {status}")

print(f"Expected per signer: {total_signs_expected}")


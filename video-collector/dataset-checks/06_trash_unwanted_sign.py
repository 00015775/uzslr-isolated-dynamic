"""
(WRITE-OUT file) USE WITH CAUTION!
Safely moves selected sign folders to the macOS Trash for all signers.

This script iterates through signer01 to signer10 and, for each sign listed in 
`signs_to_delete`, moves the corresponding sign directory (containing landmarks/videos) 
to the Trash using AppleScript via the Finder. This approach is non-destructive, 
deleted folders remain recoverable from the Trash until emptied.

Use this script when pruning unwanted or low-quality signs from the Uzbek Sign Language 
dataset. Keep `signs_to_delete` updated with the exact sign folder names to remove.

Used for dataset cleaning before final processing or model training.
"""


import os
import subprocess
from pathlib import Path

# Base directory containing signer01 to signer10
base_dir = Path('../Data_Numpy_Arrays_RSL_UzSL')

# List of sign words to delete (move to Trash)
signs_to_delete = {
    #"sign-to-delete"
}

# Signer folders: signer01 through signer10
signer_folders = [f"signer{i:02d}" for i in range(1, 11)] 

deleted_count = 0

for signer in signer_folders:
    signer_path = base_dir / signer
    if not signer_path.exists():
        print(f"Warning: {signer_path} does not exist. Skipping.")
        continue

    for sign in signs_to_delete:
        sign_folder = signer_path / sign
        if sign_folder.exists() and sign_folder.is_dir():
            try:
                # Uses AppleScript to move folder to Trash (safe, recoverable)
                subprocess.run([
                    'osascript', '-e',
                    f'tell app "Finder" to move POSIX file "{sign_folder.resolve()}" to trash'
                ], check=True)
                print(f"Moved to Trash: {sign_folder}")
                deleted_count += 1
            except subprocess.CalledProcessError as e:
                print(f"Failed to move {sign_folder} to Trash: {e}")
        else:
            print(f"Not found: {sign_folder}")

print(f"\nDone. Total folders moved to Trash: {deleted_count}")
print("You can recover them from the Trash if needed.")


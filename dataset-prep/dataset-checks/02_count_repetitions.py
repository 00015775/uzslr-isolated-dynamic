"""
(READ-ONLY file)
Counts and displays the total number of repetitions (rep-X folders) for each sign 
in the 'data/' directory.

Used for verifying that all 50 signs have the expected number of samples 
"""

from pathlib import Path

DATA_DIR = Path('../../data')

def count_repetitions_per_sign():
    print("Counting repetitions for each sign...\n")
    
    if not DATA_DIR.exists():
        print(f"Error: '{DATA_DIR}' folder not found!")
        return
    
    total_reps = 0
    
    for sign_dir in sorted(DATA_DIR.iterdir()):
        if not sign_dir.is_dir():
            continue
        
        # Count folders that start with 'rep-'
        rep_folders = [d for d in sign_dir.iterdir() if d.is_dir() and d.name.startswith('rep-')]
        count = len(rep_folders)
        
        total_reps += count
        print(f"{sign_dir.name}: {count} rep")
    
    print("\n" + "="*40)
    print(f"Total number of repetition folders: {total_reps}")
    print(f"Total number of signs: {len([d for d in DATA_DIR.iterdir() if d.is_dir()])}")

if __name__ == "__main__":
    count_repetitions_per_sign()


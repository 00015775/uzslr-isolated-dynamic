"""
(READ-ONLY file)
Counts the total number of repetitions (rep- folders) for each sign across all signers.

This script scans the processed Uzbek Sign Language dataset and, for each of the 
50 cleaned signs listed in DEFAULT_SIGNS, counts how many valid repetition folders 
(rep-*) exist in the 'landmarks' directory of every signer (signer01 to signer10).

Output: A simple list showing the total repetition count for each sign across the entire dataset.
Useful for verifying dataset balance, detecting missing recordings, or summarizing data volume.
"""

import os

# Base directory containing signer01 to signer10
root_dir = '../Data_Numpy_Arrays_RSL_UzSL'

# After data cleaning and prunning, 50 signs were left. CLEAN_SIGNS
# MAKE SURE TO KEEP THIS DICTIONARY UPDATED IN CASE IF SIGNS ARE ADDED OR REMOVED
DEFAULT_SIGNS = ['assalomu_alaykum', 'bahor', 'birga', "bo'sh", 'bosh_kiyim', 'boshlanishi', 'bozor', 'eshik', 
               'futbol', 'iltimos', 'internet', 'javob', 'jismoniy_tarbiya', 'karam', 'kartoshka', 
               'kichik', 'kitob', "ko'prik", 'likopcha', 'maktab', 'mehmonxona', 'mehribon', 'metro', 
               'musiqa', "o'simlik_yog'i", "o'ynash", 'ochish', 'ot', 'ovqat_tayyorlash', 
               'oxiri', 'poezd', 'pomidor', 'qidirish', 'qish', "qo'ziqorin", 'qor', "qorong'i", 'quyon', 
               'restoran', "sariyog'", 'shokolad', 'sovun', 'stakan', 'televizor', 'tosh', 'toza',
               'turish', "yomg'ir", 'yopish', 'yordam_berish']


# Dictionary to hold counts for each sign
sign_counts = {sign: 0 for sign in DEFAULT_SIGNS}

# Loop through each signer: signer01 to signer10
for signer_num in range(1, 11):
    signer_dir = os.path.join(root_dir, f'signer{signer_num:02d}')
    
    if not os.path.exists(signer_dir):
        continue  # Skip if signer directory does not exist
    
    # Loop through each sign
    for sign in DEFAULT_SIGNS:
        sign_dir = os.path.join(signer_dir, sign)
        landmarks_dir = os.path.join(sign_dir, 'landmarks')
        
        if os.path.exists(landmarks_dir):
            # List directories starting with 'rep-'
            rep_dirs = [d for d in os.listdir(landmarks_dir) if d.startswith('rep-') and os.path.isdir(os.path.join(landmarks_dir, d))]
            sign_counts[sign] += len(rep_dirs)

print("\n======== TOTAL REPETITIONS PER SIGN ACROSS ALL SIGNERS ========\n")

# Output the results
for sign, count in sign_counts.items():
    print(f"{sign}: {count}")

print("\n======== DONE ========\n")


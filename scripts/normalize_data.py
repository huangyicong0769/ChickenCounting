#!/usr/bin/env python3

import os
import re
import sys

def normalize_filenames(directory):
    # Change to the target directory
    os.chdir(directory)

    # Regular expression pattern to match the filenames
    # This pattern handles both with and without extension prefixes
    pattern = re.compile(
        r'^(?P<base>.+?)(?:_(?P<ext1>jpg|txt))?\.rf\.[a-f0-9]+\.(?P<ext2>jpg|txt)$',
        re.IGNORECASE
    )

    # Dictionary to map base names to their corresponding files
    file_dict = {}

    for filename in os.listdir('.'):
        match = pattern.match(filename)
        if match:
            base = match.group('base')
            ext = match.group('ext2')  # The actual extension (jpg or txt)
            normalized_name = f"{base}.{ext}"
            file_dict.setdefault(normalized_name, []).append(filename)
        else:
            # Skip files that do not match the pattern
            print(f"Skipping file with unexpected format: {filename}")

    # Process each group of files
    for normalized, files in file_dict.items():
        files_sorted = sorted(files)
        keep_file = files_sorted[0]
        duplicate_files = files_sorted[1:]

        # Rename the kept file to the normalized base name if necessary
        if keep_file != normalized:
            if os.path.exists(normalized):
                print(f"Base file '{normalized}' already exists. Deleting '{keep_file}'.")
                os.remove(keep_file)
            else:
                print(f"Renaming '{keep_file}' to '{normalized}'.")
                os.rename(keep_file, normalized)

        # Delete all duplicate files
        for dup in duplicate_files:
            if dup != normalized:
                print(f"Deleting duplicate file: '{dup}'.")
                os.remove(dup)

    print("Normalization complete.")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    else:
        target_dir = '.'

    normalize_filenames(target_dir)

#!/bin/bash

# Define the repository URL
REPO_URL="https://github.com/asukaa2/RVC-Cover-Maker"

# Navigate to the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# List of directories to keep.
# NOTE: After the migration to the `rvc` pip package, the vendored
# `programs/applio_code/` directory is gone. Predictor / embedder
# models now live in `assets/models/` (managed by the `rvc` package
# itself, which auto-downloads them from HuggingFace on first use).
declare -a KEEP_DIRS=("env" "logs" "audio_files" "assets/models")

# Function to check if a directory should be kept
should_keep_dir() {
    local dir_path="$1"
    for keep_dir in "${KEEP_DIRS[@]}"; do
        if [[ "$dir_path" == "$keep_dir" ]]; then
            return 0 # True, keep it
        fi
    done
    return 1 # False, don't keep it
}

# Delete all directories except the specified ones
for d in *; do
    if [ -d "$d" ]; then
        IS_KEEP=false
        for kp_dir in "${KEEP_DIRS[@]}"; do
            if [[ "$d" == "$kp_dir" ]]; then
                IS_KEEP=true
                break
            fi
        done

        if ! $IS_KEEP; then
            echo "Deleting directory $d"
            rm -rf "$d"
        fi
    fi
done


# Delete files, excluding the script itself
for f in *; do
    if [ -f "$f" ]; then
        if [ "$f" != "$(basename "$0")" ]; then # Exclude the script itself
            echo "Deleting file $f"
            rm -f "$f"
        fi
    fi
done

# Initialize a new git repository if it doesn't exist
if [ ! -d ".git" ]; then
    git init
    git remote add origin "$REPO_URL"
fi

# Fetch the latest changes from the repository
git fetch origin

# Reset the working directory to match the latest commit
git reset --hard origin/main

echo "Update complete. Press any key to continue..."
read -n 1 -s # This waits for a single key press without displaying it

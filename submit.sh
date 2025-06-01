#!/usr/bin/env bash
#
# submit.sh — submit a Kaggle competition file based on a run ID, then list submissions.
#
# Usage:
#   ./submit.sh <run_id>
#
# Example:
#   ./submit.sh erec0tvb
#

# Ensure exactly one argument is provided
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <run_id>"
  exit 1
fi

RUN_ID="$1"
COMPETITION="epfl-network-machine-learning-2025"

# Path to the CSV file: submissions/run_<run_id>.csv
FILE_PATH="submissions/run_${RUN_ID}.csv"

# Make sure the file exists before trying to submit
if [ ! -f "$FILE_PATH" ]; then
  echo "Error: File '$FILE_PATH' not found."
  exit 2
fi

# Submit to Kaggle
kaggle competitions submit \
  -c "$COMPETITION" \
  -f "$FILE_PATH" \
  -m "run_${RUN_ID}"

# Then list all submissions for that competition
sleep 1
kaggle competitions submissions -c "$COMPETITION"

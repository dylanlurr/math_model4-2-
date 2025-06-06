#!/bin/bash

# This script orchestrates the data processing by calling the Python worker
# for each file, preventing memory leaks.

echo "--- Starting Data Processing ---"

# Define the data directory
DATA_DIR="data"

# Process the training set
TRAIN_LIST_FILE="$DATA_DIR/file_list_train.txt"
TRAIN_CSV_FILE="$DATA_DIR/train_annotations.csv"
echo "Processing training data..."
# Create CSV header
echo "image_path,formula" > "$TRAIN_CSV_FILE"
# Read the file list and process each file, appending the output to the CSV
while IFS= read -r line; do
  python3 2_process_single_file.py "$line" | sed 's/<SEP>/,/g' >> "$TRAIN_CSV_FILE"
done < "$TRAIN_LIST_FILE"
echo "Finished processing training data."


# Process the validation set
VAL_LIST_FILE="$DATA_DIR/file_list_val.txt"
VAL_CSV_FILE="$DATA_DIR/val_annotations.csv"
echo "Processing validation data..."
# Create CSV header
echo "image_path,formula" > "$VAL_CSV_FILE"
# Read the file list and process each file, appending the output to the CSV
while IFS= read -r line; do
  python3 2_process_single_file.py "$line" | sed 's/<SEP>/,/g' >> "$VAL_CSV_FILE"
done < "$VAL_LIST_FILE"
echo "Finished processing validation data."


echo "--- All Data Processing Complete ---"


#!/bin/bash

# 腳本使用說明：
# ./run_batch.sh <輸入資料夾路徑> [輸出資料夾路徑 (選填)]

# 1. 檢查是否提供了輸入資料夾
if [ -z "$1" ]; then
    echo "Error: Please provide an input folder."
    echo "Usage: $0 <input_folder> [output_folder]"
    exit 1
fi

INPUT_FOLDER="$1"
# If the user does not provide a second argument, the default output folder is "results/invsr_results"
OUTPUT_FOLDER="${2:-results/invsr_results}"

# 2. Check if the input folder exists
if [ ! -d "$INPUT_FOLDER" ]; then
    echo "Error: Input folder '$INPUT_FOLDER' not found."
    exit 1
fi

# 3. 如果輸出資料夾不存在，則建立它
if [ ! -d "$OUTPUT_FOLDER" ]; then
    echo "Creating output folder: $OUTPUT_FOLDER"
    mkdir -p "$OUTPUT_FOLDER"
fi

echo "========================================"
echo "Starting processing..."
echo "Input source: $INPUT_FOLDER"
echo "Output target: $OUTPUT_FOLDER"
echo "========================================"

count=0

# 4. Loop through images in the folder
# Supported formats: jpg, jpeg, png, bmp, webp (case-insensitive)
for img_path in "$INPUT_FOLDER"/*.{jpg,jpeg,png,bmp,webp,JPG,JPEG,PNG}; do
    
    # Check if file exists (to avoid glob pattern being treated as a string when folder is empty)
    [ -e "$img_path" ] || continue

    filename=$(basename "$img_path")
    echo "[Processing] $filename"
    # Execute Python command
    python inference_invsr.py -i "$img_path" -o "$OUTPUT_FOLDER" --num_steps 5

    ((count++))
done

echo "========================================"
echo "Task completed! Processed a total of $count images."
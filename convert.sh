#!/bin/bash
#SBATCH --job-name=jsonl_convert
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err
#SBATCH --array=0-9          # Array jobs 0 to 9

# Load python environment if needed, e.g.:
# module load python/3.9

# Define input files or chunks - example: files_0.json to files_9.json
INPUT_DIR="./v3"
OUTPUT_DIR="./jsonl"
BATCH_SIZE=10000

INPUT_FILE="${INPUT_DIR}/file_${SLURM_ARRAY_TASK_ID}.json"
OUTPUT_FILE="${OUTPUT_DIR}/file_${SLURM_ARRAY_TASK_ID}.jsonl"

echo "Running job $SLURM_ARRAY_TASK_ID"
echo "Input: $INPUT_FILE"
echo "Output: $OUTPUT_FILE"

python process_json_parallel.py "$INPUT_FILE" "$OUTPUT_FILE" --batch_size $BATCH_SIZE
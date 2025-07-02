#!/bin/bash
#SBATCH --job-name=pes2o_embedding
#SBATCH --array=0-7                    # Creates 16 parallel jobs (0 to 15)
#SBATCH --ntasks=1                      # One task per job
#SBATCH --cpus-per-task=1              # Adjust based on your needs
#SBATCH --mem=64                  # Memory per job - adjust as needed
#SBATCH --time=12:00:00                # Max runtime - adjust as needed
#SBATCH --output=logs/embedding_%A_%a.out
#SBATCH --error=logs/embedding_%A_%a.err
         # Use appropriate partition
                  # Request GPU if needed
                    # Use single node per task
            # Specify GPU type if needed
                # Use entire node (optional)


# Create logs directory if it doesn't exist
mkdir -p logs

# Set your data path
datastore_raw_data_path= dataset/jsonl
num_shards=16

# The SLURM_ARRAY_TASK_ID is automatically set by SLURM for each job
echo "Starting job ${SLURM_ARRAY_TASK_ID} of ${num_shards}"
echo "Processing shard ${SLURM_ARRAY_TASK_ID}"

# Run the embedding task for this specific shard
PYTHONPATH=. python ric/main_ric.py \
  --config-name=pes2o_v3 \
  tasks.datastore.embedding=true \
  tasks.datastore.index=true \
  datastore.raw_data_path=$datastore_raw_data_path \
  datastore.embedding.num_shards=$num_shards \
  datastore.embedding.shard_ids=[$SLURM_ARRAY_TASK_ID] \
  hydra.job_logging.handlers.file.filename=embedding_${SLURM_ARRAY_TASK_ID}.log

echo "Completed job ${SLURM_ARRAY_TASK_ID}"
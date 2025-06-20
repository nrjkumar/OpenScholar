from huggingface_hub import snapshot_download
# snapshot_download(
#     repo_id="allenai/peS2o",
#     repo_type ="dataset",
#     allow_patterns="*.zst",
#    #cache_dir="'/network/projects/living-review'"
#    local_dir = "$SLURMTMPDIR/dataset")

from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="allenai/peS2o",
    repo_type ="dataset",
    allow_patterns="*.zst",
   #cache_dir="'/network/projects/living-review'"
   local_dir = "$SLURMTMPDIR/dataset")

from datasets import load_dataset

ds = load_dataset("OpenSciLM/OpenScholar-DataStore-V3")


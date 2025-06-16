from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="allenai/peS2o",
    repo_type ="dataset",
   allow_patterns="*.zst",
   cache_dir="/network/projects/living-review")

from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="allenai/peS2o",
    allow_patterns="/data/v3/*",
    repo_type="dataset",
    cache_dir = "/network/projects/living-review"
)
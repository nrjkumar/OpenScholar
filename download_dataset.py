from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="allenai/peS2o",
    subfolder = "v3",
    repo_type="dataset",
    cache_dir = "/network/projects/living-review",
    filename="train-*"
)
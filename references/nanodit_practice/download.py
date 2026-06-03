from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="sayakpaul/butteflies_with_classes",
    repo_type="dataset",
    local_dir="butterflies",
)

import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

# 1. download
local_dir = Path(__file__).resolve().parent / "datasets" / "butterflies"
snapshot_download(
    repo_id="sayakpaul/butteflies_with_classes",
    repo_type="dataset",
    local_dir=local_dir,
)


# 2. clean .cache
cache_dir = local_dir / ".cache"
if cache_dir.exists():
    shutil.rmtree(cache_dir)

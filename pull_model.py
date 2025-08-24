import os
from modelscope.hub.snapshot_download import snapshot_download

model_id = "qwen/Qwen-2.5-VL-3B-Instruct"
download_path = os.path.abspath(os.path.join(__file__, "../../"))
local_model_path = snapshot_download(
    model_id, 
    cache_dir=download_path
)

print(f"------------ Model downloaded to {local_model_path} ------------")
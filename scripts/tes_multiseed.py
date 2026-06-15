import aegnn
import torch
import lightning.pytorch as pl
from pathlib import Path


import os
os.environ["AEGNN_DATA_DIR"] = "/home/ibrahim/event_graph_aegnn/test_codes/data"
print(os.environ["AEGNN_DATA_DIR"])

device = "cuda" if torch.cuda.is_available() else "cpu"

model_paths = "/home/jovyan/training_log_aegnn_multiseed/checkpoints/ncars/recognition/"
root = Path(model_paths)
models = [str(p) for p in root.rglob("*.ckpt")]
print(len(models))

# data_module = aegnn.datasets.NCars(batch_size=16, shuffle=False, )
# data_module.setup()
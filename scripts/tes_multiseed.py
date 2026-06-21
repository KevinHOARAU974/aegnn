import aegnn
import torch
import lightning.pytorch as pl
from pathlib import Path
import lightning.pytorch as pl


import os

os.environ["AEGNN_DATA_DIR"] = "/home/ibrahim/event_graph_aegnn/test_codes/data"
print(os.environ["AEGNN_DATA_DIR"])

device = "cuda" if torch.cuda.is_available() else "cpu"

data_module = aegnn.datasets.NCars(
    batch_size=16,
    shuffle=False,
)
data_module.setup()
test_loader = data_module.test_dataloader()
trainer = pl.Trainer()
test_loader = data_module.test_dataloader()



model_paths = "/home/ibrahim/event_graph_aegnn/external/training_log_aegnn_multiseed/checkpoints/ncars/recognition/"
root = Path(model_paths)
models_list = [str(p) for p in root.rglob("*.ckpt")]
print(len(models_list))

for model_path in models_list:
    model = aegnn.models.recognition.RecognitionModel.load_from_checkpoint(
        model_path,
        network="graph_res",
        dataset="ncars",
        num_classes=data_module.num_classes,
        img_shape=data_module.dims,
        bias=True,
        root_weight=True,
    )
    
    results = trainer.test(model, dataloaders=test_loader)
    print(results)
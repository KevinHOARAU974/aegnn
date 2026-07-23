import yaml
import subprocess
from pathlib import Path

import lightning.pytorch as pl
import aegnn
import torch
import os


os.environ["AEGNN_DATA_DIR"] = "/home/ibrahim/event_graph_aegnn/test_codes/data/ncars/beta_processed/adaptative_preprocessing/xytp_adap_beta_1000-4-16/"
print(os.environ["AEGNN_DATA_DIR"])

config_path = Path("/home/ibrahim/event_graph_aegnn/config/config.yaml")



for seed in range(12345,12350):
    print(f"\n ------- SEED {seed} --------\n")

    #Opening the config file to change the seed
    with open(config_path,"r") as f:
        cfg = yaml.safe_load(f)

    cfg["seed"] = seed  #Changing the seed in the config file

    #Saving the seed in the config file
    with open(config_path,"w") as f:
        yaml.safe_dump(cfg,f)

    #Launch the training for the corresponding seed
    subprocess.run(["python3",
                    "/home/ibrahim/event_graph_aegnn/external/aegnn/scripts/train.py",
                    "--config",
                    str(config_path)])


# files = [
#     str(f)
#     for f in checkpoint_path.rglob("*")
#     if f.is_file() and f.name.startswith("best")
# ]
# print(files)

# trainer = pl.Trainer()
# data_module = aegnn.datasets.NCars(
#     batch_size=16,
#     shuffle=False,
# )
# data_module.setup()
# test_loader = data_module.test_dataloader()


# test_acc_list = []
# for model_path in files:
#     model = aegnn.models.recognition.RecognitionModel.load_from_checkpoint(
#         model_path,
#         network="graph_res",
#         dataset="ncars",
#         num_classes=data_module.num_classes,
#         img_shape=data_module.dims,
#         bias=True,
#         root_weight=True,
#     )
    
#     results = trainer.test(model, dataloaders=test_loader)
#     test_acc_list.append(results[0]["Test/Accuracy"])


# csv_path = "/home/ibrahim/event_graph_aegnn/training_log_aegnn_multiseed_msub/acc_list_meansub.csv"


# with open("config/config.yaml", "r") as f:
#     cfg = yaml.safe_load(f)

# batch_size = cfg["data_params"]["batch_size"]
# num_workers = cfg["data_params"]["num_workers"]
# shuffle = cfg["data_params"]["shuffle"]
# dataset = cfg["dataset"]
# lr = cfg["model_params"]["lr"]
# weight_decay = cfg["model_params"]["weight_decay"]

# with open(csv_path, "a") as f:
#     f.write(f"batch_size,{batch_size}\n")
#     f.write(f"num_workers,{num_workers}\n")
#     f.write(f"shuffle,{shuffle}\n")
#     f.write(f"dataset,{dataset}\n")
#     f.write(f"lr,{lr}\n")
#     f.write(f"weight_decay,{weight_decay}\n")
#     f.write("test acc,"+",".join(map(str, test_acc_list)))
#     f.write("\n")




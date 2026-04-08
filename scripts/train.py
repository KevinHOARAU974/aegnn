import aegnn
import wandb
import datetime
import lightning.pytorch as pl
import torch
import argparse
import yaml

from pathlib import Path
from lightning.pytorch.loggers import WandbLogger

import os

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    print(cfg)
    
    os.environ["AEGNN_DATA_DIR"] = cfg["data_dir"]
    print(os.environ["AEGNN_DATA_DIR"])

    gpu = True if torch.cuda.is_available() else False

    #Load data module
    data_module = aegnn.datasets.by_name(cfg['dataset']).from_cfg(cfg['data_params'])
    # print('module créé')
    data_module.setup()

    #Create model
    model = aegnn.models.by_task(cfg['task'])(cfg["model"],
                                      cfg["dataset"],
                                      num_classes=data_module.num_classes,
                                      img_shape=data_module.dims,
                                      max_epochs = cfg["trainer"]["max_epochs"],
                                      bias = True,
                                      root_weight = True, 
                                      **cfg["model_params"])
    print("model créé")

    project = f"aegnn-{cfg['dataset']}-{cfg['task']}"
    experiment_name = datetime.datetime.now().strftime("%Y%m%d%H%M%S")


    #Loggers
    wandb_logger = WandbLogger(project="aegnn",
                        #    groupe = "ncars",
                           name=f"{project}_{experiment_name}")
    
    wandb_logger.experiment.config.update(cfg)

    loggers = []
    loggers.append(wandb_logger)

    checkpoint_path = os.path.expanduser(os.path.join(cfg["log_dir"], "checkpoints", cfg["dataset"], cfg["task"], experiment_name))
    Path(checkpoint_path).mkdir(parents=True,exist_ok=True)

    callbacks = [
        pl.callbacks.LearningRateMonitor(),
        aegnn.utils.callbacks.BBoxLogger(classes=data_module.classes),
        # aegnn.utils.callbacks.PHyperLogger(args),
        aegnn.utils.callbacks.EpochLogger(),
        aegnn.utils.callbacks.FileLogger([model, model.model, data_module]),
        aegnn.utils.callbacks.FullModelCheckpoint(dirpath=checkpoint_path)
    ]

    #Training

    trainer_kwargs = {
    "accelerator":"gpu" if gpu else "cpu",
    "profiler": "simple" if cfg["profile"] else False,
    }

    trainer = pl.Trainer(logger=loggers,
                     callbacks=callbacks,
                     **cfg["trainer"],
                     **trainer_kwargs
                     )
    
    print("trainer créé")

    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    main()
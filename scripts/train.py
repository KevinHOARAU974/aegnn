import aegnn
import datetime
import lightning.pytorch as pl
import torch
import argparse
import yaml
import os

from pathlib import Path
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, TQDMProgressBar

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

    pl.seed_everything(cfg["seed"], workers=True)

    training(cfg)


def training(cfg):

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
    # print("model créé")

    project = f"{cfg['project_name']}-{cfg['dataset']}-{cfg['task']}"
    experiment_name = datetime.datetime.now().strftime("%Y%m%d%H%M%S")


    #Loggers
    wandb_logger = WandbLogger(project=cfg["project_name"],
                        #    groupe = "ncars",
                           name=f"{project}_{experiment_name}")
    
    wandb_logger.experiment.config.update(cfg)
    wandb_logger.experiment.config.update(aegnn.utils.git.get_git_info())
    wandb_logger.experiment.config.update({"git_dirty":aegnn.utils.git.is_dirty()})

    print(aegnn.utils.git.get_git_info())

    wandb_logger.experiment.define_metric("epoch")
    wandb_logger.experiment.define_metric("*", step_metric="epoch")

    loggers = []
    loggers.append(wandb_logger)

    checkpoint_path = os.path.expanduser(os.path.join(cfg["log_dir"], "checkpoints", cfg["dataset"], cfg["task"], experiment_name))
    Path(checkpoint_path).mkdir(parents=True,exist_ok=True)
    wandb_logger.experiment.config.update({"checkpoint_path": checkpoint_path})


    #Callbacks

    #Save best and last model 
    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=checkpoint_path,
    #     filename="best",
    #     monitor="Val/Accuracy",
    #     #monitor="Val/loss",
    #     mode='max',
    #     save_top_k=1,
    #     save_last=True,
    #     auto_insert_metric_name=False
    # )

    #Save best and last model, best val loss 
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename="best",
        monitor="Val/loss",
        mode='min',
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False
    )


    #Early Stopping
    early_stopping = EarlyStopping(
        monitor="Val/Accuracy",
        patience=cfg['callback_params']['patience'],
        mode='max',
        verbose=True
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    progress_bar =TQDMProgressBar(refresh_rate=1)

    callbacks = [
        checkpoint_callback,
        early_stopping,
        lr_monitor,
        progress_bar
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

    trainer.fit(model, datamodule=data_module)

    # Test the best model

    best_model = aegnn.models.recognition.RecognitionModel.load_from_checkpoint(f'{checkpoint_path}/best.ckpt',
                                                                                network=cfg["model"],
                                                                                dataset=cfg["dataset"],
                                                                                num_classes=data_module.num_classes,
                                                                                img_shape=data_module.dims,
                                                                                max_epochs = cfg["trainer"]["max_epochs"],
                                                                                bias = True,
                                                                                root_weight = True,
                                                                                log_dir = checkpoint_path, 
                                                                                **cfg["model_params"])

    trainer.test(best_model, data_module)

if __name__ == "__main__":
    main()
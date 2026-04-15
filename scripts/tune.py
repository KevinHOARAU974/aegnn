import optuna
import aegnn
import datetime
import lightning.pytorch as pl
import torch
import argparse
import yaml
import os
import copy

from pathlib import Path
from functools import partial
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, TQDMProgressBar
from optuna.integration import PyTorchLightningPruningCallback


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
def objective(trial: optuna.Trial, base_cfg):
    

    gpu = True if torch.cuda.is_available() else False

    cfg = copy.deepcopy(base_cfg)

    #Optuna
    cfg["model_params"]["lr"] = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    cfg["model_params"]["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    pl.seed_everything(cfg["seed"], workers=True)

    print(cfg)

    #Load data module
    data_module = aegnn.datasets.by_name(cfg['dataset']).from_cfg(cfg['data_params'])
    data_module.setup()

    model = aegnn.models.by_task(cfg['task'])(cfg["model"],
                                      cfg["dataset"],
                                      num_classes=data_module.num_classes,
                                      img_shape=data_module.dims,
                                      max_epochs = cfg["trainer"]["max_epochs"],
                                      bias = True,
                                      root_weight = True, 
                                      **cfg["model_params"])
    
    experiment_name = f"trial_{trial.number}"

    checkpoint_path = os.path.expanduser(os.path.join(cfg["log_dir"], "checkpoints", cfg["dataset"], cfg["task"], experiment_name))
    Path(checkpoint_path).mkdir(parents=True,exist_ok=True)

    #Callbacks

    #Save best and last model 
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor="Val/loss",
        mode='min',
        save_top_k=1,
        save_last=False,
        auto_insert_metric_name=False
    )

    #Early Stopping
    early_stopping = EarlyStopping(
        monitor="Val/loss",
        patience=5,
        mode='min',
        verbose=True
    )

    #Optuna Pruning Callback
    pruning = PyTorchLightningPruningCallback(trial, monitor="Val/Accuracy")

    callbacks = [
        checkpoint_callback,
        early_stopping,
        pruning
    ]

    trainer_kwargs = {
    "accelerator":"gpu" if gpu else "cpu",
    "profiler": "simple" if cfg["profile"] else False,
    }

    trainer = pl.Trainer(logger=False,
                     callbacks=callbacks,
                     **cfg["trainer"],
                     **trainer_kwargs
                     )
    
    trainer.fit(model, datamodule=data_module)

    metric = trainer.callback_metrics.get("Val/Accuracy")
    if metric is None:
        raise RuntimeError("Val/Accuracy not find")
    
    return float(metric.cpu().item())

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    base_cfg = load_config(args.config)

    os.environ["AEGNN_DATA_DIR"] = base_cfg["data_dir"]
    print(os.environ["AEGNN_DATA_DIR"])

    objective_fn = partial(objective, base_cfg=base_cfg)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective_fn, n_trials=1)

    print("Best trial:")
    print(study.best_trial.number)
    print(study.best_trial.value)
    print(study.best_trial.params)

    cfg = copy.deepcopy(base_cfg)
    cfg["model_params"]["lr"] = study.best_trial.params["lr"]
    cfg["model_params"]["weight_decay"] = study.best_trial.params["weight_decay"]

    project = f"aegnn-{cfg['dataset']}-{cfg['task']}-optuna"

    wandb_logger = WandbLogger(project="aegnn",
                        #    groupe = "ncars",
                           name=f"{project}_best_trial")
    
    wandb_logger.experiment.config.update(cfg)
    wandb_logger.experiment.config.update(aegnn.utils.git.get_git_info())
    wandb_logger.experiment.config.update({"git_dirty":aegnn.utils.git.is_dirty()})
import os 
import aegnn
import time
import pandas as pd
import yaml
import argparse
import lightning.pytorch as pl


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to folder",
    )
    args = parser.parse_args()

    cfg = load_config(f'{args.path}/config.yaml')

    print(cfg)

    os.environ["AEGNN_DATA_DIR"] = cfg["data_dir"]['value']
    print(os.environ["AEGNN_DATA_DIR"])

    pl.seed_everything(cfg["seed"]['value'], workers=True)

    # Load data module

    data_module = aegnn.datasets.by_name(cfg['dataset']['value']).from_cfg(cfg['data_params']['value'])
    # print('module créé')
    data_module.setup()

    model_path = f"{args.path}/best.ckpt"

    # Load the model
    model = aegnn.models.recognition.RecognitionModel.load_from_checkpoint(model_path, 
                                      cfg["model"]['value'],
                                      cfg["dataset"]['value'],
                                      num_classes=data_module.num_classes,
                                      img_shape=data_module.dims,
                                      max_epochs = cfg["trainer"]['value']["max_epochs"],
                                      bias = True,
                                      root_weight = True,
                                      log_dir = args.path,
                                      **cfg["model_params"]['value'])
    # print("model créé")

    trainer = pl.Trainer()

    ##### Accuracy + Confusion matrix
    test_loader = data_module.test_dataloader()

    results = trainer.test(model, dataloaders=test_loader)

    ##### Number of parameters

    num_params = sum(p.numel() for p in model.parameters())
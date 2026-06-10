import os 
import aegnn
import time
import pandas as pd
import yaml
import argparse
import lightning.pytorch as pl

from pathlib import Path


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder_path",
        type=str,
        required=True,
        help="Path to folder",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to dat",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to csv file",
    )
    args = parser.parse_args()

    cfg = load_config(f'{args.folder_path}/config.yaml')

    print(cfg)

    os.environ["AEGNN_DATA_DIR"] = args.data_path #cfg["data_dir"]['value']
    print(os.environ["AEGNN_DATA_DIR"])

    pl.seed_everything(cfg["seed"]['value'], workers=True)

    # Load data module

    data_module = aegnn.datasets.by_name(cfg['dataset']['value']).from_cfg(cfg['data_params']['value'])
    # print('module créé')
    data_module.setup()

    model_path = f"{args.folder_path}/best.ckpt"

    # Load the model
    model = aegnn.models.recognition.RecognitionModel.load_from_checkpoint(model_path, 
                                      network=cfg["model"]['value'],
                                      dataset=cfg["dataset"]['value'],
                                      num_classes=data_module.num_classes,
                                      img_shape=data_module.dims,
                                      max_epochs = cfg["trainer"]['value']["max_epochs"],
                                      bias = True,
                                      root_weight = True,
                                      log_dir = args.folder_path,
                                      **cfg["model_params"]['value'])
    # print("model créé")

    trainer = pl.Trainer()

    ##### Accuracy + Confusion matrix
    test_loader = data_module.test_dataloader()

    results_test = trainer.test(model, dataloaders=test_loader)[0]

    ##### Number of parameters

    num_params = sum(p.numel() for p in model.parameters())

    ### Save in csv files

    file_exists = Path(args.csv_path).exists()

    results = {
        'model_name' : cfg["model"]['value'],
        'dataset': cfg["dataset"]['value'],
        'seed': cfg["seed"]['value'],
        'test_loss': results_test["Test/loss"],
        'test_acc' : results_test["Test/Accuracy"],
        'num_params': num_params
    }

    print(results)

    df = pd.DataFrame([results])

    df.to_csv(args.csv_path, mode="a", header=not file_exists, index=False)

if __name__ == "__main__":
    main()

    
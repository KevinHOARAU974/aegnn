import argparse
import lightning.pytorch as pl
import torch
import yaml
import os

import aegnn

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    os.environ["AEGNN_DATA_DIR"] = cfg["data_dir"]
    print(os.environ["AEGNN_DATA_DIR"])

    if cfg['debug']:
        aegnn.utils.loggers.LoggingLogger(None, name="debug")

    if torch.cuda.is_available() and cfg['data_params']['num_workers'] > 1:
            torch.multiprocessing.set_start_method("spawn")
    pl.seed_everything(cfg['seed'])
    torch.random.manual_seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])

    dm_class = aegnn.datasets.by_name(cfg['dataset'])

    dm = dm_class(cfg["data_params"])
    dm.prepare_data()

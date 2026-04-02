import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback


class EpochLogger(Callback):

    def on_validation_end(self, trainer: pl.Trainer, model: pl.LightningModule) -> None:
        model.logger.log_metrics({"Epoch": model.current_epoch + 1})

import torch
import torch_geometric
import lightning.pytorch as pl
import wandb

from lightning.pytorch.loggers import WandbLogger
from torch.nn.functional import softmax
from typing import Tuple
from .networks import by_name as model_by_name
from torchmetrics.functional import accuracy
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

 
class RecognitionModel(pl.LightningModule):

    def __init__(self, network, dataset: str, num_classes, img_shape: Tuple[int, int],
                 dim: int = 3, lr= 1e-3, weight_decay = 5e-3, eta_min = 0.0, max_epochs = 100, scheduler_type = 'cosine', log_dir = None, test=None, **model_kwargs):
        
        super(RecognitionModel, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss()#label_smoothing=label_smoothing)
        self.optimizer_kwargs = {"lr":lr, "weight_decay":weight_decay}
        self.scheduler_kwargs = {"T_max":max_epochs, "eta_min":eta_min}

        self.scheduler_type = scheduler_type

        self.num_outputs = num_classes
        self.dim = dim #position and edge_attr dimension

        self.log_dir = log_dir
        self.test = test

        model_input_shape = torch.tensor(img_shape + (dim, ), device=self.device)
        self.model = model_by_name(network)(dataset, model_input_shape, num_outputs=num_classes, **model_kwargs)

    def forward(self, data: torch_geometric.data.Batch) -> torch.Tensor:
        data.pos = data.pos[:, :self.dim]
        data.edge_attr = data.edge_attr[:, :self.dim]
        return self.model.forward(data)
    

    ### Training with Pytorch Lightning

    def training_step(self, batch : torch_geometric.data.Batch, batch_idx : int) :

        outputs = self.forward(batch)
        loss = self.criterion(outputs, target=batch.y)
        batch_size = batch.num_graphs

        # y_prediction = torch.argmax(outputs, dim=-1)
        training_accuracy = accuracy(preds=outputs, target=batch.y, task="multiclass", num_classes=self.num_outputs)
        self.log("train/loss", loss, on_step=False, on_epoch=True, batch_size=batch_size,  prog_bar=True)
        self.log("train/acc", training_accuracy, on_step=False, on_epoch=True, batch_size=batch_size, prog_bar=True)
        return loss
    
    def on_train_epoch_end(self):
        lr = self.optimizers().param_groups[0]["lr"]

        self.log("lr-Adam", lr, on_step=False, on_epoch=True, logger=True)

    def validation_step(self, batch: torch_geometric.data.Batch, batch_idx: int) -> torch.Tensor:

        outputs = self.forward(batch)

        batch_size = batch.num_graphs

        self.log("val/loss", self.criterion(outputs, target=batch.y), on_step=False, on_epoch=True, batch_size=batch_size, prog_bar=True)
        self.log("val/acc", accuracy(preds=outputs, target=batch.y, task="multiclass", num_classes=self.num_outputs), on_step=False, on_epoch=True, batch_size=batch_size, prog_bar=True)
        k = min(3, self.num_outputs - 1)
        # self.log(f"val/acc_Top{k}", accuracy(preds=outputs, target=batch.y,  task="multiclass", num_classes=self.num_outputs, top_k=k), on_step=False, on_epoch=True, batch_size=batch_size, prog_bar=True)
        # return predictions 

    ### Test with Pytorch Lightning

    def on_test_start(self):
        
        self.test_preds = []
        self.test_targets = []

    def test_step(self, batch: torch_geometric.data.Batch, batch_idx: int):
        
        outputs = self.forward(batch)

        batch_size = batch.num_graphs

        preds = outputs.argmax(dim=1)

        self.test_preds.append(preds.cpu())
        self.test_targets.append(batch.y.cpu())
        
        if self.test == "acc": 
            self.log("best_acc_model/loss", self.criterion(outputs, target=batch.y), on_step=False, on_epoch=True, batch_size=batch_size, prog_bar=True)
            self.log("best_acc_model/acc", accuracy(preds=outputs, target=batch.y, task="multiclass", num_classes=self.num_outputs), on_step=False, on_epoch=True, batch_size=batch_size, prog_bar=True)
        
        elif self.test == "loss": 
            self.log("best_loss_model/loss", self.criterion(outputs, target=batch.y), on_step=False, on_epoch=True, batch_size=batch_size, prog_bar=True)
            self.log("best_loss_model/acc", accuracy(preds=outputs, target=batch.y, task="multiclass", num_classes=self.num_outputs), on_step=False, on_epoch=True, batch_size=batch_size, prog_bar=True)
        
        # k = min(3, self.num_outputs - 1)
        # self.log(f"Test/Accuracy_Top{k}", accuracy(preds=outputs, target=batch.y,  task="multiclass", num_classes=self.num_outputs, top_k=k), on_step=False, on_epoch=True, batch_size=batch_size, prog_bar=True)

        # return super().test_step(*args, **kwargs)

    # def on_test_end(self):
        
    #     preds = torch.cat(self.test_preds)
    #     targets = torch.cat(self.test_targets)

    #     cm = confusion_matrix(targets, preds)

    #     fig, ax = plt.subplots(figsize=(8, 8))

    #     disp = ConfusionMatrixDisplay(
    #         confusion_matrix=cm,
    #         # display_labels=data_module.classes
    #     )

    #     disp.plot(
    #         ax=ax,
    #         xticks_rotation=90,
    #         colorbar=True
    #     )

    #     plt.tight_layout()
    #     plt.savefig(f"{self.log_dir}/confusion_matrix.png", dpi=300)
    #     plt.close()

    #     wandb_logger = None

    #     for logger in self.trainer.loggers:
    #         if isinstance(logger, WandbLogger):
    #             wandb_logger = logger
    #             break

    #     if wandb_logger is not None:
        
    #         self.logger.experiment.log({
    #         "conf_mat": wandb.plot.confusion_matrix(
    #             probs=None,
    #             y_true=targets.numpy(),
    #             preds=preds.numpy(),
    #             # class_names=self.classes
    #         )
    #     })
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_kwargs)

        if self.scheduler_type == 'cosine':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **self.scheduler_kwargs)
        elif self.scheduler_type == 'step':
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=LRPolicy())
        
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": 1,
        },
    }

class LRPolicy(object):
    def __call__(self, epoch: int):
        if epoch < 20:
            return 1.0
        else:
            return 0.1
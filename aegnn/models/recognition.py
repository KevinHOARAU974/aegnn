import torch
import torch_geometric
import lightning.pytorch as pl

from torch.nn.functional import softmax
from typing import Tuple
from .networks import by_name as model_by_name
from torchmetrics.functional import accuracy


class RecognitionModel(pl.LightningModule):

    def __init__(self, network, dataset: str, num_classes, img_shape: Tuple[int, int],
                 dim: int = 3, lr= 1e-3, weight_decay = 5e-3, eta_min = 0.0, max_epochs = 100,  **model_kwargs):
        super(RecognitionModel, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer_kwargs = {"lr":lr, "weight_decay":weight_decay}
        self.scheduler_kwargs = {"T_max":max_epochs, "eta_min":eta_min}

        self.num_outputs = num_classes
        self.dim = dim #position and edge_attr dimension

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

        y_prediction = torch.argmax(outputs, dim=-1)
        training_accuracy = accuracy(preds=y_prediction, target=batch.y, task="multiclass", num_classes=self.num_outputs)
        self.logger.log_metrics({"Train/loss": loss, "Train/Accuracy": training_accuracy}, step=self.trainer.global_step)

        return loss
    
    def validation_step(self, batch: torch_geometric.data.Batch, batch_idx: int) -> torch.Tensor:

        outputs = self.forward(batch)
        y_prediction = torch.argmax(outputs, dim=-1)
        predictions = softmax(outputs, dim=-1)

        self.log("Val/Loss", self.criterion(outputs, target=batch.y))
        self.log("Val/Accuracy", accuracy(preds=y_prediction, target=batch.y, task="multiclass", num_classes=self.num_outputs))
        k = min(3, self.num_outputs - 1)
        self.log(f"Val/Accuracy_Top{k}", accuracy(preds=predictions, target=batch.y,  task="multiclass", num_classes=self.num_outputs))
        return predictions
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_kwargs)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **self.scheduler_kwargs)
        
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": 1,
        },
    }
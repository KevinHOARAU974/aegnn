import torch
import torch_geometric
import lightning.pytorch as pl

from torch.nn.functional import softmax
from typing import Tuple
from .networks import by_name as model_by_name
from torchmetrics.functional import accuracy

 
class RecognitionModel(pl.LightningModule):

    def __init__(self, network, dataset: str, num_classes, img_shape: Tuple[int, int],
                 dim: int = 3, lr= 1e-3, weight_decay = 5e-3, eta_min = 0.0, max_epochs = 100, scheduler_type = 'cosine',  **model_kwargs):
        
        super(RecognitionModel, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss()#label_smoothing=label_smoothing)
        self.optimizer_kwargs = {"lr":lr, "weight_decay":weight_decay}
        self.scheduler_kwargs = {"T_max":max_epochs, "eta_min":eta_min}

        self.scheduler_type = scheduler_type

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
        batch_size = batch.num_graphs

        # y_prediction = torch.argmax(outputs, dim=-1)
        training_accuracy = accuracy(preds=outputs, target=batch.y, task="multiclass", num_classes=self.num_outputs)
        self.log("Train/loss", loss, on_step=False, on_epoch=True, batch_size=batch_size,  prog_bar=True)
        self.log("Train/Accuracy", training_accuracy, on_step=False, on_epoch=True, batch_size=batch_size, prog_bar=True)
        return loss
    
    def on_train_epoch_end(self):
        lr = self.optimizers().param_groups[0]["lr"]

        self.log("lr-Adam", lr, on_step=False, on_epoch=True, logger=True)

    def validation_step(self, batch: torch_geometric.data.Batch, batch_idx: int) -> torch.Tensor:

        outputs = self.forward(batch)

        batch_size = batch.num_graphs

        self.log("Val/loss", self.criterion(outputs, target=batch.y), on_step=False, on_epoch=True, batch_size=batch_size, prog_bar=True)
        self.log("Val/Accuracy", accuracy(preds=outputs, target=batch.y, task="multiclass", num_classes=self.num_outputs), on_step=False, on_epoch=True, batch_size=batch_size, prog_bar=True)
        k = min(3, self.num_outputs - 1)
        self.log(f"Val/Accuracy_Top{k}", accuracy(preds=outputs, target=batch.y,  task="multiclass", num_classes=self.num_outputs, top_k=k), on_step=False, on_epoch=True, batch_size=batch_size, prog_bar=True)
        # return predictions 

    ### Test with Pytorch Lightning

    def test_step(self, batch: torch_geometric.data.Batch, batch_idx: int):
        
        outputs = self.forward(batch)

        batch_size = batch.num_graphs

        self.log("Test/loss", self.criterion(outputs, target=batch.y), on_step=False, on_epoch=True, batch_size=batch_size, prog_bar=True)
        self.log("Test/Accuracy", accuracy(preds=outputs, target=batch.y, task="multiclass", num_classes=self.num_outputs), on_step=False, on_epoch=True, batch_size=batch_size, prog_bar=True)
        k = min(3, self.num_outputs - 1)
        self.log(f"Test/Accuracy_Top{k}", accuracy(preds=outputs, target=batch.y,  task="multiclass", num_classes=self.num_outputs, top_k=k), on_step=False, on_epoch=True, batch_size=batch_size, prog_bar=True)

        # return super().test_step(*args, **kwargs)
    
    
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
import torch

from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn.pool import max_pool, voxel_grid
from typing import Callable, List, Optional, Tuple, Union


class MaxPooling(torch.nn.Module):

    def __init__(self, size: List[int], transform: Callable[[Data, ], Data] = None, start: Optional[Union[float, List[float], Tensor]] = None, end: Optional[Union[float, List[float], Tensor]] = None):
        super(MaxPooling, self).__init__()
        self.voxel_size = list(size)
        self.transform = transform
        self.start = start
        self.end = end

    def forward(self, x: torch.Tensor, pos: torch.Tensor, batch: Optional[torch.Tensor] = None,
                edge_index: Optional[torch.Tensor] = None, return_data_obj: bool = False
                ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.LongTensor, torch.Tensor, torch.Tensor], Data]:
        assert edge_index is not None, "edge_index must not be None"

        if batch is not None:
            batch = batch.long()

        if torch.is_tensor(self.voxel_size):
            self.voxel_size = self.voxel_size.to(device=pos.device, dtype=pos.dtype)
        
        if torch.is_tensor(self.start):
            self.start = self.start.to(device=pos.device, dtype=pos.dtype)

        if torch.is_tensor(self.end):
            self.end = self.end.to(device=pos.device, dtype=pos.dtype)

        cluster = voxel_grid(pos[:, :2], batch=batch, size=self.voxel_size, start=self.start, end= self.end)
        data = Data(x=x, pos=pos, edge_index=edge_index, batch=batch)
        data = max_pool(cluster, data=data, transform=self.transform)  # transform for new edge attributes
        if return_data_obj:
            return data
        else:
            return data.x, data.pos, getattr(data, "batch"), data.edge_index, data.edge_attr

    def __repr__(self):
        return f"{self.__class__.__name__}(voxel_size={self.voxel_size})"

import torch

from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn.pool import max_pool_x, voxel_grid
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from typing import List, Optional, Tuple, Union


class MaxPoolingX(torch.nn.Module):

    def __init__(self, voxel_size: List[int], size: int, start: Optional[Union[float, List[float], Tensor]] = None, end: Optional[Union[float, List[float], Tensor]] = None):
        super(MaxPoolingX, self).__init__()
        self.voxel_size = voxel_size
        self.size = size
        self.start = start
        self.end = end

    def forward(self, x: torch.Tensor, pos: torch.Tensor, batch: Optional[torch.Tensor] = None
                ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.LongTensor, torch.Tensor, torch.Tensor], Data]:
        
        pos = pos.float()

        if batch is not None:
            batch = batch.long()

        voxel_size = self.voxel_size
        start = self.start
        end = self.end
        if torch.is_tensor(voxel_size):
            voxel_size = voxel_size.to(device=pos.device, dtype=pos.dtype)
        
        if torch.is_tensor(start):
            start = self.start.to(device=pos.device, dtype=pos.dtype)

        if torch.is_tensor(end):
            end = self.end.to(device=pos.device, dtype=pos.dtype)
        # print("+-+-+-+-end",end)
        cluster = voxel_grid(pos, batch=batch, size=voxel_size, start=start, end=end)
       
        # # Number of occupied clusters
        # print("occupied clusters:", cluster.unique().numel())

        # # Number of nodes in each cluster
        # counts = torch.bincount(cluster)
        # print("cluster counts:", counts)
        # print("empty clusters:", (counts == 0).sum())

        x, _ = max_pool_x(cluster, x, batch, size=self.size)

        

        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(voxel_size={self.voxel_size}, size={self.size})"

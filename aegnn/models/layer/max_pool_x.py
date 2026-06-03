import torch

from torch_geometric.data import Data
from torch_geometric.nn.pool import max_pool_x, voxel_grid
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from typing import List, Optional, Tuple, Union


class MaxPoolingX(torch.nn.Module):

    def __init__(self, voxel_size: List[int], size: int):
        super(MaxPoolingX, self).__init__()
        self.voxel_size = voxel_size
        self.size = size

    def forward(self, x: torch.Tensor, pos: torch.Tensor, batch: Optional[torch.Tensor] = None
                ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.LongTensor, torch.Tensor, torch.Tensor], Data]:
        
        pos = pos.float()

        if batch is not None:
            batch = batch.long()
        
        size = self.voxel_size
        if torch.is_tensor(size):
            size = size.to(device=pos.device, dtype=pos.dtype)
        elif isinstance(size, (list,tuple)):
            size = [float(v) for v in size]

        cluster = voxel_grid(pos, batch=batch, size=size)

        max_index = 0

        if batch is not None:

            cluster_fixed = torch.empty_like(cluster)

            for gid in batch.unique():

                mask = batch == gid
                c_local, _ = consecutive_cluster(cluster[mask])

                cluster_fixed[mask] = c_local + max_index

                max_index += c_local.max().item() + 1
            
            cluster = cluster_fixed
        
        else:
            cluster, _ = consecutive_cluster(cluster)

        x, _ = max_pool_x(cluster, x, batch, size=self.size)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(voxel_size={self.voxel_size}, size={self.size})"

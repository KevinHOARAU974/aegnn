import glob
import numpy as np
import os
import torch

from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
from torch_geometric.nn.pool import radius_graph
from typing import Callable, List, Optional, Union

from .utils.normalization import normalize_time
from .ncaltech101 import NCaltech101


class NCars(NCaltech101):

    def __init__(
        self,
        batch_size: int = 64,
        shuffle: bool = True,
        num_workers: int = 8,
        pin_memory: bool = False,
        transform: Optional[Callable[[Data], Data]] = None,
    ):
        super(NCars, self).__init__(
            batch_size, shuffle, num_workers, pin_memory=pin_memory, transform=transform
        )
        self.dims = (120, 100)  # overwrite image shape
        pre_processing_params = {
            "r": 3.0,
            "d_max": 32,
            "n_samples": 10000,
            "sampling": True,
        }
        self.save_hyperparameters({"preprocessing": pre_processing_params})

    def read_annotations(self, raw_file: str) -> Optional[np.ndarray]:
        return None

    @staticmethod
    def read_label(raw_file: str) -> Optional[Union[str, List[str]]]:
        label_file = os.path.join(raw_file, "is_car.txt")
        with open(label_file, "r") as f:
            label_txt = f.read().replace(" ", "").replace("\n", "")
        return "car" if label_txt == "1" else "background"

    @staticmethod
    def load(raw_file: str) -> Data:
        events_file = os.path.join(raw_file, "events.txt")
        events = torch.from_numpy(np.loadtxt(events_file)).float()  # .cuda()
        x, pos = events[:, :4], events[:, :3]

        return Data(x=x, pos=pos)

    def pre_transform(self, data: Data) -> Data:
        params = self.hparams.preprocessing

        # Re-weight temporal vs. spatial dimensions to account for different resolutions.
        data.pos[:, 2] = normalize_time(data.pos[:, 2])

        # Coarsen graph by uniformly sampling n points from the event point cloud.
        data = self.sub_sampling(data, n_samples=params["n_samples"], sub_sample=params["sampling"])

        # # Radius graph generation.
        # data.edge_index = radius_graph(data.pos, r=params["r"], max_num_neighbors=params["d_max"])

        pos = data.pos.to("cuda")


        eps = 1e-9

        pos_min = pos.min(dim=0).values
        pos_max = pos.max(dim=0).values
        pos_tmp = (pos-pos_min)/(pos_max - pos_min + eps)

        N = pos.size(0)

        k_min = 2
        k_max = 10

        # [N, N] pairwise distances
        dist = torch.cdist(pos_tmp, pos_tmp)

        # ignore self-distance
        dist.fill_diagonal_(float("inf"))

        # get k_max nearest neighbors once
        dist_knn, idx_knn = torch.topk(dist, k_max, largest=False, dim=1)

        # density using first k_min neighbors
        d_i = dist_knn[:,:k_min].mean(dim=1)
        rho = 1.0 / (d_i + eps)

        # normalize rho to [0, 1]
        rho_hat = (rho - rho.min()) / (rho.max() - rho.min() + eps)

        # adaptive k for each node
        k_i = torch.floor(k_min + (k_max - k_min) * rho_hat).long()
        k_i = torch.clamp(k_i, min=k_min, max=k_max)

        # create mask: keep first k_i neighbors for each node
        cols = torch.arange(k_max, device=pos.device).unsqueeze(0)  # [1, k_max]
        mask = cols < k_i.unsqueeze(1)  # [N, k_max]

        src = torch.arange(N, device=pos.device).unsqueeze(1).expand(N, k_max)

        edge_index = torch.stack([src[mask], idx_knn[mask]], dim=0)

        edge_index = to_undirected(edge_index, num_nodes=N)
        data.edge_index = edge_index

        assert data.edge_index.min() >= 0
        assert data.edge_index.max() < data.x.shape[0]

        return data

    #########################################################################################################
    # Files #################################################################################################
    #########################################################################################################
    def raw_files(self, mode: str) -> List[str]:
        lep = os.path.expanduser(os.path.join(self.root, mode, "*"))
        # print(f"je vaius chercher ici: {lep}")
        return glob.glob(os.path.join(self.root, mode, "*"))

    def processed_files(self, mode: str) -> List[str]:
        processed_dir = os.path.expanduser(os.path.join(self.root, "processed"))
        # print(f'processed_dir : {processed_dir}')
        return glob.glob(os.path.join(processed_dir, mode, "*"))

    @property
    def classes(self) -> List[str]:
        return ["car", "background"]

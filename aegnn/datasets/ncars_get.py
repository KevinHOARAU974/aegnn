import glob
import numpy as np
import os
import torch

from torch_geometric.data import Data
from torch_geometric.nn.pool import radius_graph
from typing import Callable, List, Optional, Union

from .utils.normalization import normalize_time
from .ncaltech101 import NCaltech101
from .event_token import E2SRC_Module


class NCars_get(NCaltech101):

    def __init__(
        self,
        batch_size: int = 64,
        shuffle: bool = True,
        num_workers: int = 8,
        pin_memory: bool = False,
        transform: Optional[Callable[[Data], Data]] = None,
        method: str = 'test',
        k_min: int = 0,
        k_max: int = 32,
        k_density: int = 5,
        r: float = 3,
        d_max: int = 32,
        n_samples: int = 10000,
        sampling: bool = True,
        beta:float = 5e-7,
        dims: tuple = (120,100),
        group_num: int = 4,
        patch_size: tuple = (4,4),


    ):
        super(NCars_get, self).__init__(
            batch_size, shuffle, num_workers, pin_memory=pin_memory, transform=transform
        )
        self.method = method
        self.k_min = k_min
        self.k_max = k_max
        self.k_density = k_density
        self.r = r
        self.d_max = d_max
        self.n_samples = n_samples
        self.sampling = sampling
        self.beta = beta
        self.dims = dims  # overwrite image shape
        self.group_num = group_num
        self.patch_size = patch_size

    def read_annotations(self, raw_file: str) -> Optional[np.ndarray]:
        return None

    @staticmethod
    def read_label(raw_file: str) -> Optional[Union[str, List[str]]]:
        label_file = os.path.join(raw_file, "is_car.txt")
        with open(label_file, "r") as f:
            label_txt = f.read().replace(" ", "").replace("\n", "")
        return "car" if label_txt == "1" else "background"

    def load(self, raw_file: str) -> Data:
        events_file = os.path.join(raw_file, "events.txt")
        events = torch.from_numpy(np.loadtxt(events_file)).float()  # .cuda()
        x, pos = events[:], events[:, :3]

        e2src = E2SRC_Module(shape=self.dims, group_num=self.group_num, patch_size=self.patch_size)

        patch_tokens = e2src.get_token(x)
        x_new =  torch.concat((x,patch_tokens),dim=1) # x[N,4] -> x[N,4+token_size]
       
        return Data(x=x_new, pos=pos)
        

    def pre_transform(self, data: Data) -> Data:

        # Re-weight temporal vs. spatial dimensions to account for different resolutions.
        # data.pos[:, 2] = normalize_time(data.pos[:, 2], beta=1000) # to have max_t=100
        data.pos[:, 2] = normalize_time(data.pos[:, 2]) # to have max_t=100
        # Coarsen graph by uniformly sampling n points from the event point cloud.
        data = self.sub_sampling(data, n_samples=self.n_samples, sub_sample=self.sampling)

        if self.method == 'raduis':
                    # Radius graph generation.
                    data.edge_index = radius_graph(data.pos, r=self.r,max_num_neighbors=self.d_max)
        elif self.method == 'adaptative':
            k_min = self.k_min
            k_max = self.k_max
            k_density = self.k_density

            pos = data.pos.to("cuda")
            eps = 1e-5
            N = pos.size(0)
    

            dist = torch.cdist(pos, pos)
            dist.fill_diagonal_(float("inf"))

            dist_knn, idx_knn = torch.topk(dist, k_max, largest=False, dim=1)

            # sanity: no self-loops
            src_expand = torch.arange(N, device=pos.device).unsqueeze(1).expand(N, k_max)
            assert (idx_knn == src_expand).sum() == 0

            d_i = dist_knn[:, :k_density].mean(dim=1)
            rho = 1.0 / (d_i + eps)
            rho_hat = (rho - rho.min()) / (rho.max() - rho.min() + eps)
            k_i = torch.floor(k_min + (k_max - k_min) * rho_hat).long()
            k_i = torch.clamp(k_i, min=k_min, max=k_max)

            cols = torch.arange(k_max, device=pos.device).unsqueeze(0)
            mask = cols < k_i.unsqueeze(1)
            src = torch.arange(N, device=pos.device).unsqueeze(1).expand(N, k_max)
            edge_index = torch.stack([src[mask], idx_knn[mask]], dim=0)

            data.edge_index = edge_index
            data.num_nodes = N  # ← pin explicitly so PyG never has to infer it

            assert data.pos.shape[0] == data.x.shape[0]
            assert data.edge_index.min() >= 0
            assert data.edge_index.max() < data.num_nodes
        else:
            raise Exception("Unknown graph construction method (methods available: 'raduis', 'adaptative')")

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

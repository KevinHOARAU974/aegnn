import torch
import torch_geometric

from torch.nn import Linear
from torch.nn.functional import elu
from torch_geometric.nn.conv import SplineConv
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.transforms import Cartesian
from torch_geometric.data import Data


from aegnn.models.layer import MaxPooling, MaxPoolingX


class GraphRes(torch.nn.Module):

    def __init__(self, dataset, input_shape: torch.Tensor, num_outputs: int, pooling_size=(16, 12),
                 bias: bool = False, root_weight: bool = False):
        super(GraphRes, self).__init__()
        assert len(input_shape) == 3, "invalid input shape, should be (img_width, img_height, dim)"
        dim = int(input_shape[-1])
        self.feature_map = None
        self.fm_to_pool = None #feature map before first max pooling layer
        self.fm_to_pool_x = None #feature map before second max pooling layer

        # Set dataset specific hyper-parameters.
        if dataset == "ncars":
            kernel_size = 2
            n = [1, 8, 16, 16, 16, 32, 32, 32, 32]
            pooling_outputs = 32
        elif dataset == "ncaltech101" or dataset == "gen1":
            kernel_size = 8
            n = [1, 16, 32, 32, 32, 128, 128, 128]
            pooling_outputs = 128
        else:
            raise NotImplementedError(f"No model parameters for dataset {dataset}")

        self.conv1 = SplineConv(n[0], n[1], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm1 = BatchNorm(in_channels=n[1])
        self.conv2 = SplineConv(n[1], n[2], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm2 = BatchNorm(in_channels=n[2])

        self.conv3 = SplineConv(n[2], n[3], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm3 = BatchNorm(in_channels=n[3])
        self.conv4 = SplineConv(n[3], n[4], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm4 = BatchNorm(in_channels=n[4])

        self.conv5 = SplineConv(n[4], n[5], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm5 = BatchNorm(in_channels=n[5])
        self.pool5 = MaxPooling(pooling_size, transform=Cartesian(norm=True, cat=False))

        self.conv6 = SplineConv(n[5], n[6], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm6 = BatchNorm(in_channels=n[6])
        self.conv7 = SplineConv(n[6], n[7], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm7 = BatchNorm(in_channels=n[7])

        self.pool7 = MaxPoolingX(input_shape[:2] // 4, size=16)
        self.fc = Linear(pooling_outputs * 16, out_features=num_outputs, bias=bias)

    def forward(self, data: torch_geometric.data.Batch) -> torch.Tensor:
        x_f = data.x.clone()
        x_f = elu(self.conv1(x_f, data.edge_index, data.edge_attr))
        x_f = self.norm1(x_f)
        x_f = elu(self.conv2(x_f, data.edge_index, data.edge_attr))
        x_f = self.norm2(x_f)

        x_sc = x_f.clone()
        x_f = elu(self.conv3(x_f, data.edge_index, data.edge_attr))
        x_f = self.norm3(x_f)
        x_f = elu(self.conv4(x_f, data.edge_index, data.edge_attr))
        x_f = self.norm4(x_f)
        x_f = x_f + x_sc

        x_f = elu(self.conv5(x_f, data.edge_index, data.edge_attr))
        x_f = self.norm5(x_f)
        self.fm_to_pool = Data(x=x_f, pos=data.pos, batch=data.batch, edge_index=data.edge_index, edge_attr=data.edge_attr)
        data_pooled = self.pool5(x_f, pos=data.pos, batch=data.batch, edge_index=data.edge_index, return_data_obj=True)

        x_f = data_pooled.x.clone()
        x_sc = x_f.clone()
        x_f = elu(self.conv6(x_f, data_pooled.edge_index, data_pooled.edge_attr))
        x_f = self.norm6(x_f)
        x_f = elu(self.conv7(x_f, data_pooled.edge_index, data_pooled.edge_attr))
        x_f = self.norm7(x_f)
        x_f = x_f + x_sc

        self.fm_to_pool_x = Data(x=x_f, pos=data_pooled.pos, batch=data_pooled.batch, edge_index=data_pooled.edge_index, edge_attr=data_pooled.edge_attr)
        x = self.pool7(x_f, pos=data_pooled.pos[:, :2], batch=data_pooled.batch)
        x = x.view(-1, self.fc.in_features)
        self.feature_map = x
        return self.fc(x)

"""Implementation of PointNetVLAD model."""
from __future__ import print_function

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Variable


class NetVLADLoupe(nn.Module):
    """NetVLAD aggregation layer with gating mechanism."""

    def __init__(
        self,
        feature_size: int,
        max_samples: int,
        cluster_size: int,
        output_dim: int,
        gating: bool = True,
        add_batch_norm: bool = True,
        is_training: bool = True,
    ) -> None:
        """Initialize NetVLADLoupe layer."""
        super().__init__()
        self.feature_size = feature_size
        self.max_samples = max_samples
        self.output_dim = output_dim
        self.is_training = is_training
        self.gating = gating
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size
        self.softmax = nn.Softmax(dim=-1)
        self.cluster_weights = nn.Parameter(
            torch.randn(feature_size, cluster_size) * 1 / math.sqrt(feature_size)
        )
        self.cluster_weights2 = nn.Parameter(
            torch.randn(1, feature_size, cluster_size) * 1 / math.sqrt(feature_size)
        )
        self.hidden1_weights = nn.Parameter(
            torch.randn(cluster_size * feature_size, output_dim) * 1 / math.sqrt(feature_size)
        )

        if add_batch_norm:
            self.cluster_biases = None
            self.bn1 = nn.BatchNorm1d(cluster_size)
        else:
            self.cluster_biases = nn.Parameter(torch.randn(cluster_size) * 1 / math.sqrt(feature_size))
            self.bn1 = None

        self.bn2 = nn.BatchNorm1d(output_dim)

        if gating:
            self.context_gating = GatingContext(output_dim, add_batch_norm=add_batch_norm)

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        x = x.transpose(1, 3).contiguous()
        x = x.view((-1, self.max_samples, self.feature_size))
        activation = torch.matmul(x, self.cluster_weights)
        if self.add_batch_norm:
            # activation = activation.transpose(1,2).contiguous()
            activation = activation.view(-1, self.cluster_size)
            activation = self.bn1(activation)
            activation = activation.view(-1, self.max_samples, self.cluster_size)
            # activation = activation.transpose(1,2).contiguous()
        else:
            activation = activation + self.cluster_biases
        activation = self.softmax(activation)
        activation = activation.view((-1, self.max_samples, self.cluster_size))

        a_sum = activation.sum(-2, keepdim=True)
        a = a_sum * self.cluster_weights2

        activation = torch.transpose(activation, 2, 1)
        x = x.view((-1, self.max_samples, self.feature_size))
        vlad = torch.matmul(activation, x)
        vlad = torch.transpose(vlad, 2, 1)
        vlad = vlad - a

        vlad = F.normalize(vlad, dim=1, p=2)
        vlad = vlad.view((-1, self.cluster_size * self.feature_size))
        vlad = F.normalize(vlad, dim=1, p=2)

        vlad = torch.matmul(vlad, self.hidden1_weights)

        vlad = self.bn2(vlad)

        if self.gating:
            vlad = self.context_gating(vlad)

        return vlad


class GatingContext(nn.Module):
    """Gating context layer."""

    def __init__(self, dim: int, add_batch_norm: bool = True) -> None:
        """Initialize GatingContext layer."""
        super().__init__()
        self.dim = dim
        self.add_batch_norm = add_batch_norm
        self.gating_weights = nn.Parameter(torch.randn(dim, dim) * 1 / math.sqrt(dim))
        self.sigmoid = nn.Sigmoid()

        if add_batch_norm:
            self.gating_biases = None
            self.bn1 = nn.BatchNorm1d(dim)
        else:
            self.gating_biases = nn.Parameter(torch.randn(dim) * 1 / math.sqrt(dim))
            self.bn1 = None

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        gates = torch.matmul(x, self.gating_weights)

        if self.add_batch_norm:
            gates = self.bn1(gates)
        else:
            gates = gates + self.gating_biases

        gates = self.sigmoid(gates)

        activation = x * gates

        return activation


class Flatten(nn.Module):
    """Flatten layer."""

    def __init__(self) -> None:
        """Initialize Flatten layer."""
        super().__init__(self)

    def forward(self, input: Tensor) -> Tensor:  # noqa: D102
        return input.view(input.size(0), -1)


class STN3d(nn.Module):
    """Spatial Transformer Network for 3D data."""

    def __init__(self, num_points: int = 2500, k: int = 3, use_bn: bool = True) -> None:
        """Initialize STN3d."""
        super().__init__()
        self.k = k
        self.kernel_size = 3 if k == 3 else 1
        self.channels = 1 if k == 3 else k
        self.num_points = num_points
        self.use_bn = use_bn
        self.conv1 = torch.nn.Conv2d(self.channels, 64, (1, self.kernel_size))
        self.conv2 = torch.nn.Conv2d(64, 128, (1, 1))
        self.conv3 = torch.nn.Conv2d(128, 1024, (1, 1))
        self.mp1 = torch.nn.MaxPool2d((num_points, 1), 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.fc3.weight.data.zero_()
        self.fc3.bias.data.zero_()
        self.relu = nn.ReLU()

        if use_bn:
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(128)
            self.bn3 = nn.BatchNorm2d(1024)
            self.bn4 = nn.BatchNorm1d(512)
            self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        batchsize = x.size()[0]
        if self.use_bn:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
        else:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
        x = self.mp1(x)
        x = x.view(-1, 1024)

        if self.use_bn:
            x = F.relu(self.bn4(self.fc1(x)))
            x = F.relu(self.bn5(self.fc2(x)))
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = (
            Variable(torch.from_numpy(np.eye(self.k).astype(np.float32)))
            .view(1, self.k * self.k)
            .repeat(batchsize, 1)
        )
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetFeat(nn.Module):
    """PointNet feature extractor."""

    def __init__(
        self,
        num_points: int = 2500,
        global_feat: bool = True,
        feature_transform: bool = False,
        max_pool: bool = True,
    ) -> None:
        """Initialize PointNetFeat."""
        super().__init__()
        self.stn = STN3d(num_points=num_points, k=3, use_bn=False)
        self.feature_trans = STN3d(num_points=num_points, k=64, use_bn=False)
        self.apply_feature_trans = feature_transform
        self.conv1 = torch.nn.Conv2d(1, 64, (1, 3))
        self.conv2 = torch.nn.Conv2d(64, 64, (1, 1))
        self.conv3 = torch.nn.Conv2d(64, 64, (1, 1))
        self.conv4 = torch.nn.Conv2d(64, 128, (1, 1))
        self.conv5 = torch.nn.Conv2d(128, 1024, (1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(1024)
        self.mp1 = torch.nn.MaxPool2d((num_points, 1), 1)
        self.num_points = num_points
        self.global_feat = global_feat
        self.max_pool = max_pool

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        batchsize = x.size()[0]
        trans = self.stn(x)
        x = torch.matmul(torch.squeeze(x), trans)
        x = x.view(batchsize, 1, -1, 3)
        # x = x.transpose(2,1)
        # x = torch.bmm(x, trans)
        # x = x.transpose(2,1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        pointfeat = x
        if self.apply_feature_trans:
            f_trans = self.feature_trans(x)
            x = torch.squeeze(x)
            if batchsize == 1:
                x = torch.unsqueeze(x, 0)
            x = torch.matmul(x.transpose(1, 2), f_trans)
            x = x.transpose(1, 2).contiguous()
            x = x.view(batchsize, 64, -1, 1)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.bn5(self.conv5(x))
        if not self.max_pool:
            return x
        else:
            x = self.mp1(x)
            x = x.view(-1, 1024)
            if self.global_feat:
                return x, trans
            else:
                x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
                return torch.cat([x, pointfeat], 1), trans


class PointNetVLAD(nn.Module):
    """PointNetVLAD: Deep Point Cloud Based Retrieval for Large-Scale Place Recognition.

    Paper: https://arxiv.org/abs/1804.03492
    Code is adopted from original repository: https://github.com/mikacuy/pointnetvlad
    """

    def __init__(
        self,
        num_points: int = 2500,
        global_feat: bool = True,
        feature_transform: bool = False,
        max_pool: bool = True,
        output_dim: int = 1024,
    ) -> None:
        """Initialize PointNetVLAD model.

        Args:
            num_points (int): Number of points in the input point cloud. Defaults to 2500.
            global_feat (bool): Whether to use global feature or not. Defaults to True.
            feature_transform (bool): Whether to apply feature transform or not. Defaults to False.
            max_pool (bool): Whether to use max pooling or not. Defaults to True.
            output_dim (int): Output dimension of the model. Defaults to 1024.
        """
        super().__init__()
        self.point_net = PointNetFeat(
            num_points=num_points,
            global_feat=global_feat,
            feature_transform=feature_transform,
            max_pool=max_pool,
        )
        self.net_vlad = NetVLADLoupe(
            feature_size=1024,
            max_samples=num_points,
            cluster_size=64,
            output_dim=output_dim,
            gating=True,
            add_batch_norm=True,
            is_training=True,
        )

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        x = self.point_net(x)
        x = self.net_vlad(x)
        return x

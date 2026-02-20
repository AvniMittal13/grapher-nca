import torch
import torch.nn as nn
import torch.nn.functional as F


def pairwise_distance(x):
    """Compute pairwise distance between all nodes.
    Args:
        x: [B, N, C] node features
    Returns:
        dist: [B, N, N] pairwise squared distances
    """
    x_inner = -2 * torch.matmul(x, x.transpose(2, 1))
    x_square = torch.sum(x * x, dim=-1, keepdim=True)
    return x_square + x_inner + x_square.transpose(2, 1)


def knn(x, k):
    """Find k-nearest neighbors for each node in feature space.
    Args:
        x: [B, N, C] node features
        k: number of neighbors
    Returns:
        idx: [B, N, K] indices of k nearest neighbors
    """
    dist = pairwise_distance(x)
    _, idx = dist.topk(k=k, dim=-1, largest=False)
    return idx


def batched_index_select(x, idx):
    """Gather neighbor features using KNN indices.
    Args:
        x: [B, N, C] node features
        idx: [B, N, K] neighbor indices
    Returns:
        neighbors: [B, N, K, C] neighbor features
    """
    B, N, C = x.shape
    neighbors = x.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, C]
    idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, C)  # [B, N, K, C]
    neighbors = torch.gather(neighbors, 2, idx_expanded)  # [B, N, K, C]
    return neighbors


class MRConv2d(nn.Module):
    """Max-Relative Graph Convolution.
    Aggregation: [x_i ; max_{j in N(i)} (x_j - x_i)]
    Then linear projection with residual connection.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(in_channels * 2, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

    def forward(self, x, edge_index):
        """
        Args:
            x: [B, N, C] node features
            edge_index: [B, N, K] neighbor indices
        Returns:
            out: [B, N, C_out] updated node features
        """
        B, N, C = x.shape
        neighbors = batched_index_select(x, edge_index)  # [B, N, K, C]
        x_i = x.unsqueeze(2)  # [B, N, 1, C]
        diff = neighbors - x_i  # [B, N, K, C]
        agg = diff.max(dim=2)[0]  # [B, N, C]
        agg = torch.cat([x, agg], dim=-1)  # [B, N, 2C]

        # BatchNorm needs [B*N, 2C] -> [B*N, C_out]
        agg = agg.reshape(B * N, -1)
        out = self.nn(agg)
        out = out.reshape(B, N, -1)
        return out


class GrapherModule(nn.Module):
    """Grapher module: fc1 -> build KNN -> MRConv -> fc2 -> residual.
    Args:
        in_channels: input feature dimension
        k: number of nearest neighbors
    """
    def __init__(self, in_channels, k=9):
        super().__init__()
        self.k = k
        self.fc1 = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.BatchNorm1d(in_channels),
        )
        self.graph_conv = MRConv2d(in_channels, in_channels)
        self.fc2 = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.BatchNorm1d(in_channels),
        )

    def forward(self, x):
        """
        Args:
            x: [B, N, C] node features
        Returns:
            out: [B, N, C] updated features with residual
        """
        B, N, C = x.shape
        residual = x

        # fc1 with BatchNorm
        x = x.reshape(B * N, C)
        x = self.fc1(x)
        x = x.reshape(B, N, C)

        # Build KNN graph and apply graph convolution
        edge_index = knn(x, self.k)
        x = self.graph_conv(x, edge_index)

        # fc2 with BatchNorm
        x = x.reshape(B * N, C)
        x = self.fc2(x)
        x = x.reshape(B, N, C)

        return x + residual


class FFN(nn.Module):
    """Feed-Forward Network with expansion ratio.
    Input x -> Linear(C->4C) + BN -> GELU -> Linear(4C->C) + BN -> residual + x
    """
    def __init__(self, in_channels, expansion=4):
        super().__init__()
        hidden = in_channels * expansion
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Linear(hidden, in_channels),
            nn.BatchNorm1d(in_channels),
        )

    def forward(self, x):
        """
        Args:
            x: [B, N, C] features
        Returns:
            out: [B, N, C] with residual
        """
        B, N, C = x.shape
        residual = x
        x = x.reshape(B * N, C)
        x = self.ffn(x)
        x = x.reshape(B, N, C)
        return x + residual

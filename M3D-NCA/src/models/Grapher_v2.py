import torch
import torch.nn as nn
from src.models.Grapher import FFN  # reuse unchanged


def knn_v2(x, k, chunk_size=512):
    """Chunked KNN — O(B * chunk * N) peak memory instead of O(B * N^2).

    Uses torch.cdist (L2 distance). Since sqrt is monotonically increasing,
    argsort(L2) == argsort(L2^2), so KNN indices are identical to pairwise_distance + topk.

    Args:
        x: [B, N, C] node features
        k: number of nearest neighbors
        chunk_size: number of query rows processed per chunk
    Returns:
        idx: [B, N, K] indices of k nearest neighbors
    """
    B, N, C = x.shape
    idx = torch.empty(B, N, k, dtype=torch.long, device=x.device)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        dists = torch.cdist(x[:, start:end, :], x)          # [B, chunk, N]
        _, idx[:, start:end, :] = dists.topk(k=k, dim=-1, largest=False)
    return idx


def batched_index_select_v2(x, idx):
    """Gather neighbor features using flat integer indexing.

    Avoids the virtual [B, N, N, C] expand+gather path in v1.
    Produces identical output — different memory layout only.

    Args:
        x: [B, N, C] node features
        idx: [B, N, K] neighbor indices
    Returns:
        neighbors: [B, N, K, C] neighbor features
    """
    B, N, C = x.shape
    K = idx.shape[2]
    batch_offsets = torch.arange(B, device=x.device).view(B, 1, 1) * N
    global_idx = (idx + batch_offsets).reshape(-1)           # [B*N*K]
    out = x.reshape(B * N, C)[global_idx]                    # [B*N*K, C]
    return out.view(B, N, K, C)


class MRConv2d_v2(nn.Module):
    """Max-Relative Graph Convolution (v2) — no materialised diff tensor.

    Mathematical equivalence:
        max_{j in N(i)}(x_j - x_i) == max_{j in N(i)}(x_j) - x_i
    because x_i is constant per node i across the K neighbors.

    Uses batched_index_select_v2 for coalesced memory access.
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
        neighbors = batched_index_select_v2(x, edge_index)   # [B, N, K, C]
        # max(x_j - x_i) == max(x_j) - x_i  (x_i constant per node i)
        agg = neighbors.max(dim=2)[0] - x                    # [B, N, C]
        agg = torch.cat([x, agg], dim=-1)                    # [B, N, 2C]

        agg = agg.reshape(B * N, -1)
        out = self.nn(agg)
        out = out.reshape(B, N, -1)
        return out


class GrapherModule_v2(nn.Module):
    """Grapher module v2: chunked KNN + optimised MRConv.

    Drop-in replacement for GrapherModule. Same fc1/fc2/residual structure,
    identical forward signature and output shape.

    Args:
        in_channels: input feature dimension
        k: number of nearest neighbors
        chunk_size: rows per KNN distance chunk (lower = less peak VRAM)
    """
    def __init__(self, in_channels, k=9, chunk_size=512):
        super().__init__()
        self.k = k
        self.chunk_size = chunk_size
        self.fc1 = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.BatchNorm1d(in_channels),
        )
        self.graph_conv = MRConv2d_v2(in_channels, in_channels)
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

        x = x.reshape(B * N, C)
        x = self.fc1(x)
        x = x.reshape(B, N, C)

        edge_index = knn_v2(x, self.k, chunk_size=self.chunk_size)
        x = self.graph_conv(x, edge_index)

        x = x.reshape(B * N, C)
        x = self.fc2(x)
        x = x.reshape(B, N, C)

        return x + residual

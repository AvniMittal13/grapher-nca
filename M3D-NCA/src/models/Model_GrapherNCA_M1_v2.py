import torch
import torch.nn as nn
from src.models.Model_BasicNCA import BasicNCA
from src.models.Grapher_v2 import GrapherModule_v2
from src.models.Grapher import FFN


class GrapherNCA_M1_v2(BasicNCA):
    """Pixel-Grapher-NCA v2 (m1v2): memory-optimised drop-in for GrapherNCA_M1.

    Identical interface, weights, and forward logic. Only the internal graph
    operations are optimised:
      - Chunked KNN (O(B*chunk*N) vs O(B*N^2) peak memory)
      - Flat index gather (no virtual [B,N,N,C] expand)
      - max(neighbors) - x aggregation (no [B,N,K,C] diff tensor)
    """
    def __init__(self, channel_n, fire_rate, device, hidden_size=512,
                 input_channels=1, k=9, chunk_size=512):
        super(GrapherNCA_M1_v2, self).__init__(channel_n, fire_rate, device,
                                               hidden_size, input_channels)
        self.k = k

        del self.fc0
        del self.fc1

        self.grapher = GrapherModule_v2(channel_n, k=k, chunk_size=chunk_size)
        self.ffn = FFN(channel_n, expansion=4)

        self.to(self.device)

    def update(self, x_in, fire_rate):
        """Update using Grapher module instead of Sobel perception + FC.
        Args:
            x_in: [B, H, W, C] channel-last format (NCA convention)
            fire_rate: stochastic activation rate
        """
        B, H, W, C = x_in.shape

        x = x_in.reshape(B, H * W, C)

        dx = self.grapher(x)
        dx = self.ffn(dx)
        dx = dx - x  # Extract delta (grapher/ffn have residual connections)

        dx = dx.reshape(B, H, W, C)

        if fire_rate is None:
            fire_rate = self.fire_rate
        stochastic = torch.rand([dx.size(0), dx.size(1), dx.size(2), 1]) > fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        return x_in + dx

    def forward(self, x, steps=64, fire_rate=0.5):
        """Forward: iterate update, freeze input channels.
        Args:
            x: [B, H, W, C] input with channel_n channels (first input_channels are image)
            steps: number of NCA steps
            fire_rate: stochastic activation rate
        """
        for step in range(steps):
            x2 = self.update(x, fire_rate).clone()
            x = torch.concat((x[..., :self.input_channels],
                              x2[..., self.input_channels:]), 3)
        return x

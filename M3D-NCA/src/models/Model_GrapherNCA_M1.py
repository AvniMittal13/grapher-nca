import torch
import torch.nn as nn
from src.models.Model_BasicNCA import BasicNCA
from src.models.Grapher import GrapherModule, FFN


class GrapherNCA_M1(BasicNCA):
    """Pixel-Grapher-NCA (m1): Each pixel is a graph node.
    The Grapher module replaces the Sobel/conv perception entirely.
    KNN graph is built in feature space, max-relative aggregation + FFN produces the update.
    """
    def __init__(self, channel_n, fire_rate, device, hidden_size=512,
                 input_channels=1, k=9):
        super(GrapherNCA_M1, self).__init__(channel_n, fire_rate, device,
                                            hidden_size, input_channels)
        self.k = k

        # Replace FC layers from BasicNCA with Grapher + FFN
        # Remove inherited fc0, fc1
        del self.fc0
        del self.fc1

        self.grapher = GrapherModule(channel_n, k=k)
        self.ffn = FFN(channel_n, expansion=4)

        self.to(self.device)

    def update(self, x_in, fire_rate):
        """Update using Grapher module instead of Sobel perception + FC.
        Args:
            x_in: [B, H, W, C] channel-last format (NCA convention)
            fire_rate: stochastic activation rate
        """
        B, H, W, C = x_in.shape

        # Reshape to graph: [B, N, C] where N = H*W
        x = x_in.reshape(B, H * W, C)

        # Grapher + FFN produce the update
        dx = self.grapher(x)
        dx = self.ffn(dx)
        dx = dx - x  # Extract the delta (grapher/ffn have residual connections)

        # Reshape back to spatial
        dx = dx.reshape(B, H, W, C)

        # Stochastic fire rate masking
        if fire_rate is None:
            fire_rate = self.fire_rate
        stochastic = (torch.rand([dx.size(0), dx.size(1), dx.size(2), 1],
                                 device=self.device) > fire_rate).float()
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

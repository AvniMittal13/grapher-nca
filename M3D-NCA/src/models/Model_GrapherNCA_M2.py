import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.Model_BackboneNCA import BackboneNCA
from src.models.Grapher import GrapherModule, FFN


class GrapherNCA_M2(BackboneNCA):
    """Patch-Grapher-NCA (m2): Image divided into patches, each patch = graph node.
    Grapher produces patch embeddings which are concatenated with standard conv perception.
    Hybrid local (conv) + global (graph) perception.
    """
    def __init__(self, channel_n, fire_rate, device, hidden_size=512,
                 input_channels=1, k=9, patch_size=4):
        super(GrapherNCA_M2, self).__init__(channel_n, fire_rate, device,
                                            hidden_size, input_channels)
        self.k = k
        self.patch_size = patch_size

        # Override fc0 to accept 4*channel_n (3*C from conv perception + C from grapher)
        del self.fc0
        del self.fc1
        self.fc0 = nn.Linear(channel_n * 4, hidden_size)
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)
        with torch.no_grad():
            self.fc1.weight.zero_()

        # Grapher operates on patch nodes
        self.grapher = GrapherModule(channel_n, k=k)
        self.ffn = FFN(channel_n, expansion=4)

        self.to(self.device)

    def graph_perceive(self, x):
        """Build graph on patches and broadcast back to pixel level.
        Args:
            x: [B, C, H, W] channel-first
        Returns:
            patch_features: [B, C, H, W] graph features broadcast to pixel level
        """
        B, C, H, W = x.shape
        ps = self.patch_size

        # Average pool into patches: [B, C, H/ps, W/ps]
        patches = F.avg_pool2d(x, kernel_size=ps, stride=ps)
        pH, pW = patches.shape[2], patches.shape[3]

        # Reshape to graph nodes: [B, K, C] where K = (H/ps)*(W/ps)
        patch_nodes = patches.reshape(B, C, pH * pW).transpose(1, 2)  # [B, K, C]

        # Apply Grapher + FFN
        patch_nodes = self.grapher(patch_nodes)
        patch_nodes = self.ffn(patch_nodes)

        # Reshape back to spatial: [B, C, pH, pW]
        patch_features = patch_nodes.transpose(1, 2).reshape(B, C, pH, pW)

        # Broadcast back to pixel level via nearest upsample
        patch_features = F.interpolate(patch_features, size=(H, W), mode='nearest')

        return patch_features

    def update(self, x_in, fire_rate):
        """Update with hybrid conv + graph perception.
        Args:
            x_in: [B, H, W, C] channel-last (NCA convention)
            fire_rate: stochastic activation rate
        """
        # Transpose to channel-first for conv operations
        x = x_in.transpose(1, 3)  # [B, C, H, W]

        # Standard conv perception: [B, 3C, H, W]
        conv_perception = self.perceive(x)

        # Graph perception: [B, C, H, W]
        graph_feat = self.graph_perceive(x)

        # Concatenate: [B, 4C, H, W]
        combined = torch.cat([conv_perception, graph_feat], dim=1)

        # Transpose back to channel-last and pass through FC layers
        combined = combined.transpose(1, 3)  # [B, H, W, 4C]
        dx = self.fc0(combined)
        dx = F.relu(dx)
        dx = self.fc1(dx)

        # Stochastic fire rate masking
        if fire_rate is None:
            fire_rate = self.fire_rate
        stochastic = torch.rand([dx.size(0), dx.size(1), dx.size(2), 1]) > fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        return x_in + dx

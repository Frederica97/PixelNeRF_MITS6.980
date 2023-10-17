import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
import torch


class PositionalEncoding(nn.Module):
    def __init__(self, num_octaves: int):
        super().__init__()
        self.num_octaves = num_octaves

    def forward(
        self,
        samples: Float[Tensor, "*batch dim"],
    ) -> Float[Tensor, "*batch embedded_dim"]:
        """Separately encode each channel using a positional encoding. The lowest
        frequency should be 2 * torch.pi, and each frequency thereafter should be
        double the previous frequency. For each frequency, you should encode the input
        signal using both sine and cosine.
        """
        num_batch, num_dim = samples.shape

        pe = torch.zeros(num_batch, 2 * self.num_octaves)
        for i in range(self.num_octaves):
            pe[:, 2 * i] = torch.sin(2 ** (i - 1) * torch.pi * samples[:, 0])
            pe[:, 2 * i + 1] = torch.cos(2 ** (i - 1) * torch.pi * samples[:, 1])

        return pe

    def d_out(self, dimensionality: int):
        return dimensionality * 2 * self.num_octaves

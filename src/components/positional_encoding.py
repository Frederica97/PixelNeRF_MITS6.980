import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


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

        frequencies = torch.arange(
            2 * torch.pi, 2 * self.num_octaves * torch.pi, self.num_octaves
        )
        phases = torch.tensor(torch.pi / 2)
        frequencies = frequencies.repeat(2, 1)
        phases = phases.repeat(self.num_octaves)
        zeros = torch.zeros_like(phases)
        phases = torch.stack((zeros, phases))

        pe = torch.sin(
            torch.einsum("bi, pf-> bpfi", samples, frequencies) + phases.unsqueeze(-1)
        )
        pe = torch.reshape(pe, (samples.shape[0], -1))
        return pe

    def d_out(self, dimensionality: int):
        return dimensionality * 2 * self.num_octaves

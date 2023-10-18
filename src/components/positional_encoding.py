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

        dim = len(samples[-1])
        mul = torch.zeros(dim, 2 * dim * self.num_octaves)
        for i in range(dim):
            for j in range(self.num_octaves):
                mul[i, 2 * i + 2 * j] = 2 ** (j - 1) * torch.pi
                mul[i, 2 * i + 2 * j + 1] = 2 ** (j - 1) * torch.pi
        pe = torch.einsum("...i, ij-> ...j", samples, mul)

        last_dim = -1
        num_cols = pe.size(last_dim)

        mask_even = torch.arange(num_cols) % 2 == 0
        mask_odd = ~mask_even

        pe[..., mask_even] = torch.sin(pe[..., mask_even])
        pe[..., mask_odd] = torch.cos(pe[..., mask_odd])
        return pe

        # if samples.shape[1] < 2:
        #     pe = torch.zeros(samples.shape[0], 2 * self.num_octaves)
        #     for i in range(self.num_octaves):
        #         pe[:, 2 * i] = torch.sin(2 ** (i - 1) * torch.pi * samples[:, 0])
        #         pe[:, 2 * i + 1] = torch.cos(2 ** (i - 1) * torch.pi * samples[:, 0])
        # else:
        #     num_batch, num_dim = samples.shape

        #     pe = torch.zeros(num_batch, self.d_out(len(samples.shape)))
        #     for i in range(self.num_octaves):
        #         pe[:, 4 * i] = torch.sin(2 ** (i - 1) * torch.pi * samples[:, 0])
        #         pe[:, 4 * i + 1] = torch.cos(2 ** (i - 1) * torch.pi * samples[:, 0])
        #         pe[:, 4 * i + 2] = torch.sin(2 ** (i - 1) * torch.pi * samples[:, 1])
        #         pe[:, 4 * i + 3] = torch.cos(2 ** (i - 1) * torch.pi * samples[:, 1])

    def d_out(self, dimensionality: int):
        return dimensionality * 2 * self.num_octaves

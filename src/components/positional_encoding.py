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
        batch_size, sequence_length, num_channels = samples.shape
        position = torch.arange(sequence_length).unsqueeze(0).repeat(batch_size, 1)
        div_term = torch.exp(
            torch.arange(0, num_channels, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / num_channels)
        )

        sinusoids = torch.sin(position.float() / div_term)
        cosinuses = torch.cos(position.float() / div_term)

        # Repeat the positional encodings for each example in the batch
        sinusoids = sinusoids.unsqueeze(0).repeat(batch_size, 1, 1)
        cosinuses = cosinuses.unsqueeze(0).repeat(batch_size, 1, 1)

        # Combine sinusoids and cosinuses to create positional encodings
        encodings = torch.cat([sinusoids, cosinuses], dim=-1)

        # Add positional encodings to the input samples
        samples_with_pe = samples + encodings

        return samples_with_pe

    def d_out(self, dimensionality: int):
        return dimensionality + 2 * self.num_octaves

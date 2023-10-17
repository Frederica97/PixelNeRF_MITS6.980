import torch.nn as nn
from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor

from .field import Field
from src.components.positional_encoding import PositionalEncoding


class FieldMLP(Field):
    def __init__(
        self,
        cfg: DictConfig,
        d_coordinate: int,
        d_out: int,
    ) -> None:
        """Set up an MLP for the neural field. Your architecture must respect the
        following parameters from the configuration (in config/field/mlp.yaml):

        - positional_encoding_octaves: The number of octaves in the positional encoding.
          If this parameter is None, do not positionally encode the input.
        - num_hidden_layers: The number of hidden linear layers.
        - d_hidden: The dimensionality of the hidden layers.

        Don't forget to add ReLU between your linear layers!
        """

        super().__init__(cfg, d_coordinate, d_out)

        self.num_octaves = cfg.positional_encoding_octaves
        num_hidden_layers = cfg.num_hidden_layers
        d_hidden = cfg.d_hidden

        layers = []
        input_dim = d_coordinate

        if self.num_octaves is not None:
            self.pe = PositionalEncoding(self.num_octaves)
            input_dim = self.num_octaves * 2

        layers.append(nn.Linear(input_dim, d_hidden))
        layers.append(nn.ReLU())

        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(d_hidden, d_hidden))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(d_hidden, d_out))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        coordinates: Float[Tensor, "batch coordinate_dim"],
    ) -> Float[Tensor, "batch output_dim"]:
        """Evaluate the MLP at the specified coordinates."""

        if self.num_octaves is not None:
            coordinates = self.pe(coordinates)
        return self.mlp(coordinates)

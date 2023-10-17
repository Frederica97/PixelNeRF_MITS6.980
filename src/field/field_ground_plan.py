import torch
from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor

from src.components.positional_encoding import PositionalEncoding
from src.field.field_grid import FieldGrid
from src.field.field_mlp import FieldMLP

from .field import Field


class FieldGroundPlan(Field):
    def __init__(
        self,
        cfg: DictConfig,
        d_coordinate: int,
        d_out: int,
    ) -> None:
        """Set up a neural ground plan. You should reuse the following components:

        - FieldGrid from  src/field/field_grid.py
        - FieldMLP from src/field/field_mlp.py
        - PositionalEncoding from src/components/positional_encoding.py

        Your ground plan only has to handle the 3D case.
        """
        super().__init__(cfg, d_coordinate, d_out)
        # assert d_coordinate == 3, "d_coordinate should be equal to 3"
        assert d_coordinate in (2, 3)

        self.pe = PositionalEncoding(cfg.positional_encoding_octaves)
        self.grid = FieldGrid(cfg.grid, d_coordinate, cfg.d_grid_feature)
        self.mlp = FieldMLP(
            cfg.mlp, cfg.d_grid_feature + 2 * cfg.positional_encoding_octaves, d_out
        )

    def forward(
        self,
        coordinates: Float[Tensor, "batch coordinate_dim"],
    ) -> Float[Tensor, "batch output_dim"]:
        """Evaluate the ground plan at the specified coordinates. You should:

        - Sample the grid using the X and Y coordinates.
        - Positionally encode the Z coordinates.
        - Concatenate the grid's outputs with the corresponding encoded Z values, then
          feed the result through the MLP.
        """
        coordinates_xy = coordinates[:, 0:2]
        coordinates_z = coordinates[:, 1:]

        feature_z = self.pe(coordinates_z)
        feature_xy = self.grid(coordinates_xy)

        features = torch.cat((feature_xy, feature_z), 1)

        return self.mlp(features)

from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor

from .field import Field
import torch.nn as nn
import torch


class FieldGrid(Field):
    def __init__(
        self,
        cfg: DictConfig,
        d_coordinate: int,
        d_out: int,
    ) -> None:
        """Set up a grid for the neural field. Your architecture must respect the
        following parameters from the configuration (in config/field/grid.yaml):

        - side_length: the side length in each dimension

        Your architecture only needs to support 2D and 3D grids.
        """
        super().__init__(cfg, d_coordinate, d_out)
        assert d_coordinate in (2, 3)

        grid = torch.tensor([0])
        side_length = cfg.side_length

        if d_coordinate == 2:
            grid = torch.zeros(d_out, side_length, side_length)

        else:
            grid = torch.zeros(d_out, side_length, side_length, side_length)

        self.grid = nn.Parameter(grid)
        self.d_coordinate = d_coordinate

    def forward(
        self,
        coordinates: Float[Tensor, "batch coordinate_dim"],
    ) -> Float[Tensor, "batch output_dim"]:
        """Use torch.nn.functional.grid_sample to bilinearly sample from the image grid.
        Remember that your implementation must support either 2D and 3D queries,
        depending on what d_coordinate was during initialization.
        """

        coordinates = coordinates * 2 - 1
        if self.d_coordinate == 2:
            coordinates = coordinates.unsqueeze(1).unsqueeze(1)
            grid = self.grid.unsqueeze(0).expand(coordinates.shape[0], -1, -1, -1)

        if self.d_coordinate == 3:
            coordinates = coordinates.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            grid = self.grid.unsqueeze(0).expand(coordinates.shape[0], -1, -1, -1, -1)

        return nn.functional.grid_sample(grid, coordinates).squeeze(-1).squeeze(-1)

from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor

from src.field.field_grid import FieldGrid
from src.field.field_mlp import FieldMLP

from .field import Field


class FieldHybridGrid(Field):
    def __init__(
        self,
        cfg: DictConfig,
        d_coordinate: int,
        d_out: int,
    ) -> None:
        """Set up a hybrid grid-mlp neural field. You should reuse FieldGrid from
        src/field/field_grid.py and FieldMLP from src/field/field_mlp.py in your
        implementation.

        Hint: Since you're reusing existing components, you only need to add one line
        each to __init__ and forward!
        """
        super().__init__(cfg, d_coordinate, d_out)

        self.grid = FieldGrid(cfg.grid, d_coordinate, cfg.d_grid_feature)
        self.mlp = FieldMLP(cfg.mlp, cfg.d_grid_feature, d_out)

    def forward(
        self,
        coordinates: Float[Tensor, "batch coordinate_dim"],
    ) -> Float[Tensor, "batch output_dim"]:
        return self.mlp(self.grid(coordinates))

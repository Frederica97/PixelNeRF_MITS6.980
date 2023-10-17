import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float
from omegaconf import DictConfig
from PIL import Image
from torch import Tensor

from .field_dataset import FieldDataset


class FieldDatasetImage(FieldDataset):
    def __init__(self, cfg: DictConfig) -> None:
        """Load the image in cfg.path into memory here."""

        super().__init__(cfg)
        image = Image.open(cfg.path)
        image_array = np.array(image)
        self.image = torch.from_numpy(image_array)
        self.image = torch.tensor(self.image, dtype=torch.float32)

    def query(
        self,
        coordinates: Float[Tensor, "batch d_coordinate"],
    ) -> Float[Tensor, "batch d_out"]:
        """Sample the image at the specified coordinates and return the corresponding
        colors. Remember that the coordinates will be in the range [0, 1].

        You may find the grid_sample function from torch.nn.functional helpful here.
        Pay special attention to grid_sample's expected input range for the grid
        parameter.
        """
        batch_size = coordinates.shape[0]
        image = (
            self.image.unsqueeze(0).expand(batch_size, -1, -1, -1).permute(0, 3, 1, 2)
        )
        coordinates = coordinates * 2 - 1
        coordinates = coordinates.unsqueeze(1).unsqueeze(1)
        sampled_colors = nn.functional.grid_sample(image, coordinates)
        return sampled_colors.squeeze(-1).squeeze(-1) / 255

    @property
    def d_coordinate(self) -> int:
        return 2

    @property
    def d_out(self) -> int:
        return 3

    @property
    def grid_size(self) -> tuple[int, ...]:
        """Return a grid size that corresponds to the image's shape."""
        return tuple(self.image.shape[:2])

import torch
from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor, nn

from .field.field import Field


class NeRF(nn.Module):
    cfg: DictConfig
    field: Field

    def __init__(self, cfg: DictConfig, field: Field) -> None:
        super().__init__()
        self.cfg = cfg
        self.field = field

    def forward(
        self,
        origins: Float[Tensor, "batch 3"],
        directions: Float[Tensor, "batch 3"],
        near: float,
        far: float,
    ) -> Float[Tensor, "batch 3"]:
        """Render the rays using volumetric rendering. Use the following steps:

        1. Generate sample locations along the rays using self.generate_samples().
        2. Evaluate the neural field at the sample locations. The neural field's output
           has four channels: three for RGB color and one for volumetric density. Don't
           forget to map these channels to valid output ranges.
        3. Compute the alpha values for the evaluated volumetric densities using
           self.compute_alpha_values().
        4. Composite these alpha values together with the evaluated colors from.
        """

        sample_locations, boundaries = self.generate_samples(
            origins, directions, near, far, self.cfg.num_samples
        )

        field_output = [self.field(e) for e in sample_locations]
        field_output = torch.stack(field_output)

        rgb_color, density = torch.split(field_output, [3, 1], dim=-1)
        rgb_color = torch.sigmoid(rgb_color)
        density = nn.ReLU()(density)

        alpha = self.compute_alpha_values(density.squeeze(), boundaries)
        final_image = self.alpha_composite(alpha, rgb_color)
        return final_image

    def generate_samples(
        self,
        origins: Float[Tensor, "batch 3"],
        directions: Float[Tensor, "batch 3"],
        near: float,
        far: float,
        num_samples: int,
    ) -> tuple[
        Float[Tensor, "batch sample 3"],  # xyz sample locations
        Float[Tensor, "batch sample+1"],  # sample boundaries
    ]:
        """For each ray, equally divide the space between the specified near and far
        planes into num_samples segments. Return the segment boundaries (including the
        endpoints at the near and far planes). Also return sample locations, which fall
        at the midpoints of the segments.
        """
        num_batch = origins.shape[0]
        boundaries = torch.linspace(near, far, num_samples + 1)
        midpoints = boundaries[:-1] + boundaries[1:] / 2

        samples = torch.einsum(" i , j k-> j i k", midpoints, directions)
        locations = origins.unsqueeze(1).expand(-1, num_samples, -1) + samples
        boundaries = boundaries.unsqueeze(0).expand(num_batch, -1)

        return locations, boundaries

    def compute_alpha_values(
        self,
        sigma: Float[Tensor, "batch sample"],
        boundaries: Float[Tensor, "batch sample+1"],
    ) -> Float[Tensor, "batch sample"]:
        """Compute alpha values from volumetric densities (values of sigma) and segment
        boundaries.
        """
        delta_values = torch.diff(boundaries)
        alpha_values = 1.0 - torch.exp(-sigma * delta_values)
        return alpha_values

    def alpha_composite(
        self,
        alphas: Float[Tensor, "batch sample"],
        colors: Float[Tensor, "batch sample 3"],
    ) -> Float[Tensor, "batch 3"]:
        """Alpha-composite the supplied alpha values and colors. You may assume that the
        background is black.
        """
        T = torch.ones_like(alphas)
        for i in range(1, alphas.shape[1]):
            T[:, i] = torch.prod(1 - alphas[:, :i], dim=1)
        weights = T * alphas
        colors = torch.sum(weights.unsqueeze(-1) * colors, dim=1)  # sum over samples
        return colors

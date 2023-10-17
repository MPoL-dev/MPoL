from __future__ import annotations

from typing import Protocol

import mpol.coordinates
import mpol.fourier
import mpol.images


class MPoLModel(Protocol):
    coords: mpol.coordinates.GridCoords
    nchan: int
    bcube: mpol.images.BaseCube
    icube: mpol.images.ImageCube
    fcube: mpol.fourier.FourierCube

    def forward(self):
        ...

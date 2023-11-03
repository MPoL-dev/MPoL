import torch

from mpol.coordinates import GridCoords

from . import fourier, images


class SimpleNet(torch.nn.Module):
    r"""
    A basic but fully functional network for RML imaging.

    Args:
        cell_size (float): the width of a pixel [arcseconds]
        npix (int): the number of pixels per image side
        coords (GridCoords): an object already instantiated from the GridCoords class. If providing this, cannot provide ``cell_size`` or ``npix``.
        nchan (int): the number of channels in the base cube. Default = 1.
        base_cube : a pre-packed base cube to initialize the model with. If None, assumes ``torch.zeros``.

    After the object is initialized, instance variables can be accessed, for example

    :ivar bcube: the :class:`~mpol.images.BaseCube` instance
    :ivar icube: the :class:`~mpol.images.ImageCube` instance
    :ivar fcube: the :class:`~mpol.fourier.FourierCube` instance

    For example, you'll likely want to access the ``self.icube.sky_model`` at some point.

    The idea is that :class:`~mpol.precomposed.SimpleNet` can save you some keystrokes composing models by connecting the most commonly used layers together.

    .. mermaid:: _static/mmd/src/SimpleNet.mmd

    """

    def __init__(
        self,
        coords=None,
        nchan=1,
        base_cube=None,
    ):
        super().__init__()

        self.coords = coords
        self.nchan = nchan

        self.bcube = images.BaseCube(
            coords=self.coords, nchan=self.nchan, base_cube=base_cube
        )

        self.conv_layer = images.HannConvCube(nchan=self.nchan)

        self.icube = images.ImageCube(
            coords=self.coords, nchan=self.nchan, passthrough=True
        )
        self.fcube = fourier.FourierCube(coords=self.coords)

    @classmethod
    def from_image_properties(cls, cell_size, npix, nchan, base_cube):
        coords = GridCoords(cell_size, npix)
        return cls(coords, nchan, base_cube)

    def forward(self):
        r"""
        Feed forward to calculate the model visibilities. In this step, a :class:`~mpol.images.BaseCube` is fed to a :class:`~mpol.images.HannConvCube` is fed to a :class:`~mpol.images.ImageCube` is fed to a :class:`~mpol.fourier.FourierCube` to produce the visibility cube.

        Returns: 1D complex torch tensor of model visibilities.
        """
        x = self.bcube()
        x = self.conv_layer(x)
        x = self.icube(x)
        vis = self.fcube(x)
        return vis

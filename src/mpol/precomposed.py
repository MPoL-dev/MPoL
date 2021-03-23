from . import images
from . import connectors
from .coordinates import _setup_coords
import torch


class SimpleNet(torch.nn.Module):
    r"""
    A basic but fully functional network for RML imaging. 

    Args:
        cell_size (float): the width of a pixel [arcseconds]
        npix (int): the number of pixels per image side
        coords (GridCoords): an object already instantiated from the GridCoords class. If providing this, cannot provide ``cell_size`` or ``npix``.
        nchan (int): the number of channels in the base cube. Default = 1.
        griddedDataset: instantiated :class:`~mpol.datasets.GriddedDataset` object
        base_cube : a pre-packed base cube to initialize the model with. If None, assumes ``torch.zeros``.
    
    After the object is initialized, instance variables can be accessed, for example
    
    :ivar bcube: the :class:`~mpol.images.BaseCube` instance
    :ivar icube: the :class:`~mpol.images.ImageCube` instance

    For example, you'll likely want to access the ``self.icube.sky_model`` at some point.

    The idea is that :class:`~mpol.precomposed.SimpleNet` can save you some keystrokes composing models by connecting the most commonly used layers together.
    
    .. mermaid:: _static/mmd/src/SimpleNet.mmd
    
    """

    def __init__(
        self,
        cell_size=None,
        npix=None,
        coords=None,
        nchan=None,
        griddedDataset=None,
        base_cube=None,
    ):
        super().__init__()

        _setup_coords(self, cell_size, npix, coords, nchan)

        self.bcube = images.BaseCube(
            coords=self.coords, nchan=self.nchan, base_cube=base_cube
        )
        self.icube = images.ImageCube(
            coords=self.coords, nchan=self.nchan, passthrough=True
        )
        self.flayer = images.FourierCube(coords=self.coords)

        assert griddedDataset is not None, "Please provide a GriddedDataset instance."
        self.dcon = connectors.GriddedDatasetConnector(self.flayer, griddedDataset)

    def forward(self):
        r"""
        Feed forward to calculate the model visibilities. In this step, a :class:`~mpol.images.BaseCube` is fed to a :class:`~mpol.images.ImageCube` is fed to a :class:`~mpol.images.FourierCube` is fed to a :class:`~mpol.connectors.DatasetConnector` to calculate the model visibilities at the indexed locations of the :class:`~mpol.datasets.GriddedDataset` object that was used to instantiate the class.

        Returns: 1D complex torch tensor of model visibilities.
        """
        x = self.bcube.forward()
        x = self.icube.forward(x)
        vis = self.flayer.forward(x)
        model_samples = self.dcon.forward(vis)
        return model_samples

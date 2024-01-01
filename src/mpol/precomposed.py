import torch

from mpol.coordinates import GridCoords

from mpol import fourier
from mpol import images


class GriddedNet(torch.nn.Module):
    r"""
    A basic but functional network for RML imaging. Designed to optimize a model image
    using the entirety of the dataset in a :class:`mpol.datasets.GriddedDataset`
    (i.e., gradient descent). For stochastic gradient descent (SGD), where the model
    is only seeing a fraction of the dataset with each iteration, we recommend defining
    your own module in your analysis code, following the 'Getting Started' guide.


    .. note:

        This module is provided as a starting point. However, we recommend
        that you *don't get too comfortable using it*. Instead, we recommend that you
        *start writing your own* (custom) module following PyTorch idioms, potentially
        using the source of this routine as a reference point to get started. Using
        the torch module building system directly is *much more powerful* and will
        cultivate good machine learning model building habits.

    .. mermaid:: _static/mmd/src/GriddedNet.mmd

    Parameters
    ----------
    coords : :class:`mpol.coordinates.GridCoords`
    nchan : int
        the number of channels in the base cube. Default = 1.
    base_cube : :class:`mpol.images.BaseCube` or ``None``
        a pre-packed base cube to initialize the model with. If
        None, assumes ``torch.zeros``.

    After the object is initialized, instance variables can be accessed, for example

    :ivar bcube: the :class:`~mpol.images.BaseCube` instance
    :ivar icube: the :class:`~mpol.images.ImageCube` instance
    :ivar fcube: the :class:`~mpol.fourier.FourierCube` instance

    For example, you'll likely want to access the ``self.icube.sky_model``
    at some point.
    """

    def __init__(
        self,
        coords: GridCoords,
        nchan: int = 1,
        base_cube: torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        self.coords = coords
        self.nchan = nchan

        self.bcube = images.BaseCube(
            coords=self.coords, nchan=self.nchan, base_cube=base_cube
        )

        self.conv_layer = images.HannConvCube(nchan=self.nchan)

        self.icube = images.ImageCube(coords=self.coords, nchan=self.nchan)
        self.fcube = fourier.FourierCube(coords=self.coords)
        self.nufft = fourier.NuFFT(coords=self.coords, nchan=self.nchan)

    def forward(self) -> torch.Tensor:
        r"""
        Feed forward to calculate the model visibilities. In this step, a
        :class:`~mpol.images.BaseCube` is fed to a :class:`~mpol.images.HannConvCube`
        is fed to a :class:`~mpol.images.ImageCube` is fed to a
        :class:`~mpol.fourier.FourierCube` to produce the visibility cube.

        Returns
        -------
            1D complex torch tensor of model visibilities.
        """
        x = self.bcube()
        x = self.conv_layer(x)
        x = self.icube(x)
        vis: torch.Tensor = self.fcube(x)
        return vis

    def predict_loose_visibilities(
        self, uu: torch.Tensor, vv: torch.Tensor
    ) -> torch.Tensor:
        """
        Use the :class:`mpol.fourier.NuFFT` to calculate loose model visibilities from
        the cube stored to ``self.icube.cube``.

        Parameters
        ----------
        uu : :class:`torch.Tensor` of `class:`torch.double`
            array of u spatial frequency coordinates,
            not including Hermitian pairs. Units of [:math:`\mathrm{k}\lambda`]
        vv : :class:`torch.Tensor` of `class:`torch.double`
            array of v spatial frequency coordinates,
            not including Hermitian pairs. Units of [:math:`\mathrm{k}\lambda`]

        Returns
        -------
        :class:`torch.Tensor` of `class:`torch.complex128`
            model visibilities corresponding to ``uu`` and ``vv`` locations.
        """

        model_vis: torch.Tensor = self.nufft(self.icube.cube, uu, vv)
        return model_vis

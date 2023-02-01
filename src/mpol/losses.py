r"""The following loss functions are available to use in imaging. Many of the definitions follow those in Appendix A of `EHT-IV 2019 <https://ui.adsabs.harvard.edu/abs/2019ApJ...875L...4E/abstract>`_, including the regularization strength, which aspires to be similar across all terms, providing at least a starting point for tuning multiple loss functions.

If you don't see a loss function you need, it's easy to write your own directly within your optimization script. If you like it, please consider opening a pull request!
"""

import numpy as np
import torch

from . import connectors
from .constants import *


def chi_squared(model_vis, data_vis, weight):
    r"""
    Compute the :math:`\chi^2` between the complex data :math:`\boldsymbol{V}` and model :math:`M` visibilities using

    .. math::

        \chi^2(\boldsymbol{V}|\,\boldsymbol{\theta}) = \sum_i^N \frac{|V_i - M(u_i, v_i |\,\boldsymbol{\theta})|^2}{\sigma_i^2}

    where :math:`\sigma_i^2 = 1/w_i`. The sum is over all of the provided visibilities. This function is agnostic as to whether the sum should include the Hermitian conjugate visibilities, but be aware that the answer returned will be different between the two cases. We recommend not including the Hermitian conjugates.

    Args:
        model_vis (PyTorch complex): array tuple of the model representing :math:`\boldsymbol{V}`
        data_vis (PyTorch complex): array of the data values representing :math:`M`
        weight (PyTorch real): array of weight values representing :math:`w_i`

    Returns:
        torch.double: the :math:`\chi^2` likelihood
    """
    # print("inside chi_squared")
    # print("model", model_vis.shape)
    # print("data", data_vis.shape)
    # print("weight", weight.shape)

    return torch.sum(weight * torch.abs(data_vis - model_vis) ** 2)


def log_likelihood(model_vis, data_vis, weight):
    r"""
    Compute the log likelihood function :math:`\ln\mathcal{L}` between the complex data :math:`\boldsymbol{V}` and model :math:`M` visibilities using

    .. math::

        \ln \mathcal{L}(\boldsymbol{V}|\,\boldsymbol{\theta}) = - \left ( N \ln 2 \pi +  \sum_i^N \sigma_i^2 + \frac{1}{2} \chi^2(\boldsymbol{V}|\,\boldsymbol{\theta}) \right )

    where :math:`\chi^2` is evaluated using :func:`mpol.losses.chi_squared`.

    This function is agnostic as to whether the sum should include the Hermitian conjugate visibilities, but be aware that the normalization of the answer returned will be different between the two cases. Inference of the parameter values should be unaffected. We recommend not including the Hermitian conjugates.

    Args:
        model_vis (PyTorch complex): array tuple of the model representing :math:`\boldsymbol{V}`
        data_vis (PyTorch complex): array of the data values representing :math:`M`
        weight (PyTorch real): array of weight values representing :math:`w_i`

    Returns:
        torch.double: the :math:`\ln\mathcal{L}` log likelihood
    """

    # If model and data are multidimensional, then flatten them to get full N
    N = len(torch.ravel(data_vis))

    sigma_term = torch.sum(1 / weight)

    return (
        N * np.log(2 * np.pi)
        + sigma_term
        + 0.5 * chi_squared(model_vis, data_vis, weight)
    )


def nll(model_vis, data_vis, weight):
    r"""
    Calculate a normalized "negative log likelihood" loss between the complex data :math:`\boldsymbol{V}` and model :math:`M` visibilities using

    .. math::

        L_\mathrm{nll} = \frac{1}{2 N} \chi^2(\boldsymbol{V}|\,\boldsymbol{\theta})

    where :math:`\chi^2` is evaluated using :func:`mpol.losses.chi_squared`. Visibilities may be any shape as long as all quantities have the same shape. Following `EHT-IV 2019 <https://ui.adsabs.harvard.edu/abs/2019ApJ...875L...4E/abstract>`_, we apply
    a prefactor :math:`1/(2 N)`, where :math:`N` is the number of visibilities. The factor of 2 comes in because we must count real and imaginaries in the :math:`\chi^2` sum. This means that this normalized negative log likelihood loss function will have a minimum value of $L_\mathrm{nll}(\hat{\boldsymbol{\theta}}) \approx 1$ for a well-fit model (regardless of the number of data points), making it easier to set the prefactor strengths of other regularizers *relative* to this value.

    Note that this function should only be used in an optimization or point estimate situation. If it is used in any situation where uncertainties on parameter values are determined (such as Markov Chain Monte Carlo), it will return the wrong answer. This is because the relative scaling of :math:`L_\mathrm{nll}` with respect to parameter value is incorrect.

    Args:
        model_vis (PyTorch complex): array tuple of the model representing :math:`\boldsymbol{V}`
        data_vis (PyTorch complex): array of the data values representing :math:`M`
        weight (PyTorch real): array of weight values representing :math:`w_i`

    Returns:
        torch.double: the normalized negative log likelihood likelihood loss
    """

    # If model and data are multidimensional, then flatten them to get full N
    N = len(torch.ravel(data_vis))

    return 1 / (2 * N) * chi_squared(model_vis, data_vis, weight)


def chi_squared_gridded(vis, griddedDataset):
    r"""
    Calculate the :math:`\chi^2` (corresponding to :func:`~mpol.losses.chi_squared`) using gridded data and model visibilities.

    Args:
        vis (torch complex tensor): torch tensor with shape ``(nchan, npix, npix)`` to be indexed by the ``mask`` from :class:`~mpol.datasets.GriddedDataset`. Assumes tensor is "pre-packed," as in output from :meth:`mpol.fourier.FourierCube.forward()`.
        griddedDataset: instantiated :class:`~mpol.datasets.GriddedDataset` object

    Returns:
        torch.double: the :math:`\chi^2` value

    """

    # use the index connector to get the model_visibilities from the dataset
    # 1D torch tensor collapsed across cube dimensions, like
    # griddedDataset.vis_indexed and griddedDataset.weight_indexed
    model_vis = connectors.index_vis(vis, griddedDataset)

    return chi_squared(
        model_vis, griddedDataset.vis_indexed, griddedDataset.weight_indexed
    )


def log_likelihood_gridded(vis, griddedDataset):
    r"""
    Calculate the log likelihood function :math:`\ln\mathcal{L}` (corresponding to :func:`~mpol.losses.log_likelihood`) using gridded data and model visibilities.

    Args:
        vis (torch complex tensor): torch tensor with shape ``(nchan, npix, npix)`` to be indexed by the ``mask`` from :class:`~mpol.datasets.GriddedDataset`. Assumes tensor is "pre-packed," as in output from :meth:`mpol.fourier.FourierCube.forward()`.
        griddedDataset: instantiated :class:`~mpol.datasets.GriddedDataset` object

    Returns:
        torch.double: the :math:`\ln\mathcal{L}` value

    """

    # use the index connector to get the model_visibilities from the dataset
    # 1D torch tensor collapsed across cube dimensions, like
    # griddedDataset.vis_indexed and griddedDataset.weight_indexed
    model_vis = connectors.index_vis(vis, griddedDataset)

    return log_likelihood(
        model_vis, griddedDataset.vis_indexed, griddedDataset.weight_indexed
    )


def nll_gridded(vis, datasetGridded):
    r"""
    Calculate a normalized "negative log likelihood" (corresponding to :func:`~mpol.losses.nll`) using gridded data and model visibilities. Function will return the same value regardless of whether Hermitian pairs are included.

    Args:
        vis (torch complex tensor): torch tensor with shape ``(nchan, npix, npix)`` to be indexed by the ``mask`` from :class:`~mpol.datasets.GriddedDataset`. Assumes tensor is "pre-packed," as in output from :meth:`mpol.fourier.FourierCube.forward()`.
        griddedDataset: instantiated :class:`~mpol.datasets.GriddedDataset` object

    Returns:
        torch.double: the normalized negative log likelihood likelihood loss
    """
    model_vis = connectors.index_vis(vis, datasetGridded)

    return nll(model_vis, datasetGridded.vis_indexed, datasetGridded.weight_indexed)


def entropy(cube, prior_intensity):
    r"""
    Calculate the entropy loss of a set of pixels following the definition in `EHT-IV 2019 <https://ui.adsabs.harvard.edu/abs/2019ApJ...875L...4E/abstract>`_.

    Args:
        cube (any tensor): pixel values must be positive :math:`I_i > 0` for all :math:`i`
        prior_intensity (any tensor): the prior value :math:`p` to calculate entropy against. Could be a single constant or an array the same shape as image.

    Returns:
        torch.double: entropy loss

    The entropy loss is calculated as

    .. math::

        L = \frac{1}{\sum_i I_i} \sum_i I_i \; \ln \frac{I_i}{p_i}
    """
    # check to make sure image is positive, otherwise raise an error
    assert (cube >= 0.0).all(), "image cube contained negative pixel values"
    assert prior_intensity > 0, "image prior intensity must be positive"

    tot = torch.sum(cube)
    return (1 / tot) * torch.sum(cube * torch.log(cube / prior_intensity))


def TV_image(sky_cube, epsilon=1e-10):
    r"""
    Calculate the total variation (TV) loss in the image dimension (R.A. and DEC). Following the definition in `EHT-IV 2019 <https://ui.adsabs.harvard.edu/abs/2019ApJ...875L...4E/abstract>`_ Promotes the image to be piecewise smooth and the gradient of the image to be sparse.

    Args:
        sky_cube (any 3D tensor): the image cube array :math:`I_{lmv}`, where :math:`l` is R.A. in :math:`ndim=3`, :math:`m` is DEC in :math:`ndim=2`, and :math:`v` is the channel (velocity or frequency) dimension in :math:`ndim=1`. Should be in sky format representation.
        epsilon (float): a softening parameter in [:math:`\mathrm{Jy}/\mathrm{arcsec}^2`]. Any pixel-to-pixel variations within each image slice greater than this parameter will have a significant penalty.

    Returns:
        torch.double: total variation loss

    .. math::

        L = \sum_{l,m,v} \sqrt{(I_{l + 1, m, v} - I_{l,m,v})^2 + (I_{l, m+1, v} - I_{l, m, v})^2 + \epsilon}

    """

    # diff the cube in ll and remove the last row
    diff_ll = sky_cube[:, 0:-1, 1:] - sky_cube[:, 0:-1, 0:-1]

    # diff the cube in mm and remove the last column
    diff_mm = sky_cube[:, 1:, 0:-1] - sky_cube[:, 0:-1, 0:-1]

    loss = torch.sum(torch.sqrt(diff_ll**2 + diff_mm**2 + epsilon))

    return loss


def TV_channel(cube, epsilon=1e-10):
    r"""
    Calculate the total variation (TV) loss in the channel dimension. Following the definition in `EHT-IV 2019 <https://ui.adsabs.harvard.edu/abs/2019ApJ...875L...4E/abstract>`_.

    Args:
        cube (any 3D tensor): the image cube array :math:`I_{lmv}`
        epsilon (float): a softening parameter in [:math:`\mathrm{Jy}/\mathrm{arcsec}^2`]. Any channel-to-channel pixel variations greater than this parameter will have a significant penalty.

    Returns:
        torch.double: total variation loss

    .. math::

        L = \sum_{l,m,v} \sqrt{(I_{l, m, v + 1} - I_{l,m,v})^2 + \epsilon}

    """
    # calculate the difference between the n+1 cube and the n cube
    diff_vel = cube[1:] - cube[0:-1]
    loss = torch.sum(torch.sqrt(diff_vel**2 + epsilon))

    return loss


def edge_clamp(cube):
    r"""
    Promote all pixels at the edge of the image to be zero using an :math:`L_2` norm.

    Args:
        cube (any 3D tensor): the array and pixel values

    Returns:
        torch.double: edge loss
    """

    # find edge pixels
    # all channels
    # pixel edges
    bt_edges = cube[:, (0, -1)]
    lr_edges = cube[:, :, (0, -1)]

    loss = torch.sum(bt_edges**2) + torch.sum(lr_edges**2)

    return loss


def sparsity(cube, mask=None):
    r"""
    Enforce a sparsity prior on the image cube using the :math:`L_1` norm. Optionally provide a boolean mask to apply the prior to only the ``True`` locations. For example, you might want this mask to be ``True`` for background regions.

    Args:
        cube (nchan, npix, npix): tensor image cube
        mask (boolean): tensor array the same shape as ``cube``. The sparsity prior will be applied to those pixels where the mask is ``True``. Default is to apply prior to all pixels.

    Returns:
        torch.double: sparsity loss calculated where ``mask == True``

    The sparsity loss calculated as

    .. math::

        L = \sum_i | I_i |
    """

    if mask is not None:
        loss = torch.sum(torch.abs(cube.masked_select(mask)))
    else:
        loss = torch.sum(torch.abs(cube))

    return loss


def UV_sparsity(vis, qs, q_max):
    r"""
    Enforce a sparsity prior for all :math:`q = \sqrt{u^2 + v^2}` points larger than :math:`q_\mathrm{max}`.

    Args:
        vis (torch.double) : visibility cube of (nchan, npix, npix//2 +1, 2)
        qs: numpy array corresponding to visibility coordinates. Dimensionality of (npix, npix//2)
        q_max (float): maximum radial baseline

    Returns:
        torch.double: UV sparsity loss above :math:`q_\mathrm{max}`
    """

    # make a mask, then send it to the device (in case we're using a GPU)
    mask = torch.tensor((qs > q_max), dtype=torch.bool).to(vis.device)

    vis_re = vis[:, :, :, 0]
    vis_im = vis[:, :, :, 1]

    # broadcast mask to the same shape as vis
    mask = mask.unsqueeze(0)

    loss = torch.sum(torch.abs(vis_re.masked_select(mask))) + torch.sum(
        torch.abs(vis_im.masked_select(mask))
    )

    return loss


def PSD(qs, psd, l):
    r"""
    Apply a loss function corresponding to the power spectral density using a Gaussian process kernel.

    Assumes an image plane kernel of

    .. math::

        k(r) = exp(-\frac{r^2}{2 \ell^2})

    The corresponding power spectral density is

    .. math::

        P(q) = (2 \pi \ell^2) exp(- 2 \pi^2 \ell^2 q^2)


    Args:
        qs (torch.double): the radial UV coordinate (in kilolambda)
        psd (torch.double): the power spectral density cube
        l (torch.double): the correlation length in the image plane (in arcsec)

    Returns:
        torch.double : the loss calculated using the power spectral density

    """

    # stack to the full 3D shape
    qs = qs * 1e3  # lambda

    l_rad = l * arcsec  # radians

    # calculate the expected power spectral density
    expected_PSD = (
        2 * np.pi * l_rad**2 * torch.exp(-2 * np.pi**2 * l_rad**2 * qs**2)
    )

    # evaluate the chi^2 for the PSD, making sure it broadcasts across all channels
    loss = torch.sum(psd / expected_PSD)

    return loss


def TSV(sky_cube):
    r"""
    Calculate the total square variation (TSV) loss in the image dimension (R.A. and DEC). Following the definition in `EHT-IV 2019 <https://ui.adsabs.harvard.edu/abs/2019ApJ...875L...4E/abstract>`_ Promotes the image to be edge smoothed which may be a better reoresentation of the truth image `K. Kuramochi et al 2018 <https://ui.adsabs.harvard.edu/abs/2018ApJ...858...56K/abstract>`_.

    Args:
        sky_cube (any 3D tensor): the image cube array :math:`I_{lmv}`, where :math:`l` is R.A. in :math:`ndim=3`, :math:`m` is DEC in :math:`ndim=2`, and :math:`v` is the channel (velocity or frequency) dimension in :math:`ndim=1`. Should be in sky format representation.

    Returns:
        torch.double: total square variation loss

    .. math::

        L = \sum_{l,m,v} (I_{l + 1, m, v} - I_{l,m,v})^2 + (I_{l, m+1, v} - I_{l, m, v})^2

    """

    # diff the cube in ll and remove the last row
    diff_ll = sky_cube[:, 0:-1, 1:] - sky_cube[:, 0:-1, 0:-1]

    # diff the cube in mm and remove the last column
    diff_mm = sky_cube[:, 1:, 0:-1] - sky_cube[:, 0:-1, 0:-1]

    loss = torch.sum(diff_ll**2 + diff_mm**2)

    return loss

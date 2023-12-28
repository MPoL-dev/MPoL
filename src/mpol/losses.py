r"""The following loss functions are available to use in imaging. Many of the 
definitions follow those in Appendix A of 
`EHT-IV 2019 <https://ui.adsabs.harvard.edu/abs/2019ApJ...875L...4E/abstract>`_, 
including the regularization strength, which aspires to be similar across all terms, 
providing at least a starting point for tuning multiple loss functions.

If you don't see a loss function you need, it's easy to write your own directly within 
your optimization script. If you like it, please consider opening a pull request!
"""

import numpy as np
import torch


from mpol import constants
from mpol.datasets import GriddedDataset
from typing import Optional


def chi_squared(
    model_vis: torch.Tensor, data_vis: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    r"""
    Computes the :math:`\chi^2` between the complex data :math:`\boldsymbol{V}` and
    model :math:`M` visibilities using

    .. math::

        \chi^2(\boldsymbol{\theta};\,\boldsymbol{V}) =
        \sum_i^N w_i |V_i - M(u_i, v_i |\,\boldsymbol{\theta})|^2

    where :math:`w_i = 1/\sigma_i^2`. The sum is over all of the provided visibilities.
    This function is agnostic as to whether the sum should include the Hermitian
    conjugate visibilities, but be aware that the answer returned will be different
    between the two cases. We recommend not including the Hermitian conjugates.

    Parameters
    ----------
    model_vis : :class:`torch.Tensor` of :class:`torch.complex`
        array of the model values representing :math:`\boldsymbol{V}`
    data_vis : :class:`torch.Tensor` of :class:`torch.complex`
        array of the data values representing :math:`M`
    weight : :class:`torch.Tensor` of :class:`torch.double`
        array of weight values representing :math:`w_i`

    Returns
    -------
    :class:`torch.Tensor` of :class:`torch.double`
        the :math:`\chi^2` likelihood, summed over all dimensions of input array.
    """

    return torch.sum(weight * torch.abs(data_vis - model_vis) ** 2)


def log_likelihood(
    model_vis: torch.Tensor, data_vis: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    r"""
    Compute the log likelihood function :math:`\ln\mathcal{L}` between the complex data
    :math:`\boldsymbol{V}` and model :math:`M` visibilities using

    .. math::

        \ln \mathcal{L}(\boldsymbol{\theta};\,\boldsymbol{V}) =
        - N \ln 2 \pi +  \sum_i^N w_i -
        \frac{1}{2} \chi^2(\boldsymbol{\theta};\,\boldsymbol{V})

    where :math:`N` is the number of complex visibilities and :math:`\chi^2` is
    evaluated using :func:`mpol.losses.chi_squared`. Note that this expression has
    factors of 2 in different places compared to the multivariate Normal you might be
    used to seeing because the visibilities are complex-valued. We could alternatively
    write

    .. math::

        \mathcal{L}(\boldsymbol{\theta};\,\boldsymbol{V}) =
        \mathcal{L}(\boldsymbol{\theta};\,\Re\{\boldsymbol{V}\}) \times
        \mathcal{L}(\boldsymbol{\theta};\,\Im\{\boldsymbol{V}\})

    where :math:`\mathcal{L}(\boldsymbol{\theta};\,\Re\{\boldsymbol{V}\})` and
    :math:`\mathcal{L}(\boldsymbol{\theta};\,\Im\{\boldsymbol{V}\})` each are the
    well-known multivariate Normal for reals.

    This function is agnostic as to whether the sum should include the Hermitian
    conjugate visibilities, but be aware that the normalization of the answer returned
    will be different between the two cases. Inference of the parameter values should
    be unaffected. We recommend not including the Hermitian conjugates.

    Parameters
    ----------
    model_vis : :class:`torch.Tensor` of :class:`torch.complex128`
        array of the model values representing :math:`\boldsymbol{V}`
    data_vis : :class:`torch.Tensor` of :class:`torch.complex128`
        array of the data values representing :math:`M`
    weight : :class:`torch.Tensor` of :class:`torch.double`
        array of weight values representing :math:`w_i`

    Returns
    -------
    :class:`torch.Tensor` of :class:`torch.double`
        the :math:`\ln\mathcal{L}` log likelihood, summed over all dimensions
        of input array.
    """

    # If model and data are multidimensional, then flatten them to get full N
    N = len(torch.ravel(data_vis))

    weight_term: torch.Tensor = torch.sum(torch.log(weight))

    # calculate separately so we can type as np, otherwise mypy thinks
    # the expression is Any
    first_term: np.float64 = -N * np.log(2 * np.pi)

    return first_term + weight_term - 0.5 * chi_squared(model_vis, data_vis, weight)


def reduced_chi_squared(
    model_vis: torch.Tensor, data_vis: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    r"""
    Calculate the reduced :math:`\chi^2_\mathrm{r}` between the complex data
    :math:`\boldsymbol{V}` and model :math:`M` visibilities using

    .. math::

        \chi^2_\mathrm{r} = \frac{1}{2 N} \chi^2(\boldsymbol{\theta};\,\boldsymbol{V})

    where :math:`\chi^2` is evaluated using :func:`mpol.losses.chi_squared`.
    Data and model visibilities may be any shape as long as all tensors (including
    weight) have the same shape. Following `EHT-IV 2019
    <https://ui.adsabs.harvard.edu/abs/2019ApJ...875L...4E/abstract>`_, we apply
    a prefactor :math:`1/(2 N)`, where :math:`N` is the number of visibilities. The
    factor of 2 comes in because we must count real and imaginaries in the
    :math:`\chi^2` sum. This means that this normalized negative log likelihood loss
    function will have a minimum value of
    :math:`\chi^2_\mathrm{r}(\hat{\boldsymbol{\theta}};\,\boldsymbol{V})
    \approx 1` for a well-fit model (regardless of the number of data points), making
    it easier to set the prefactor strengths of other regularizers *relative* to this
    value.

    Note that this function should only be used in an optimization or point estimate
    situation `and` where you are not adjusting the weight or the amplitudes of
    the data values. If it is used in any situation where uncertainties on parameter values
    are determined (such as Markov Chain Monte Carlo), it will return the wrong answer.
    This is because the relative scaling of :math:`\chi^2_\mathrm{r}` with respect to
    parameter value is incorrect. For those applications, you should use
    :meth:`mpol.losses.log_likelihood`.

    Parameters
    ----------
    model_vis : :class:`torch.Tensor` of :class:`torch.complex`
        array of the model values representing :math:`\boldsymbol{V}`
    data_vis : :class:`torch.Tensor` of :class:`torch.complex`
        array of the data values representing :math:`M`
    weight : :class:`torch.Tensor` of :class:`torch.double`
        array of weight values representing :math:`w_i`

    Returns
    -------
    :class:`torch.Tensor` of :class:`torch.double`
        the :math:`\chi^2_\mathrm{r}`, summed over all dimensions of input array.
    """

    # If model and data are multidimensional, then flatten them to get full N
    N = len(torch.ravel(data_vis))

    return 1 / (2 * N) * chi_squared(model_vis, data_vis, weight)


def neg_log_likelihood_avg(
    model_vis: torch.Tensor, data_vis: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    r"""
    Calculate the average value of the negative log likelihood

    .. math::

        - \frac{1}{2 N} \ln \mathcal{L}(\boldsymbol{\theta};\,\boldsymbol{V})

    where :math:`N` is the number of complex visibilities. This loss function is most
    useful where you are in an optimization or point estimate
    situation `and` where you may adjusting the weight or the amplitudes of
    the data values, perhaps via a self-calibration operation.

    If you are in any situation where uncertainties on parameter values
    are determined (such as Markov Chain Monte Carlo), you should use
    :meth:`mpol.losses.log_likelihood`.

    Parameters
    ----------
    model_vis : :class:`torch.Tensor` of :class:`torch.complex`
        array of the model values representing :math:`\boldsymbol{V}`
    data_vis : :class:`torch.Tensor` of :class:`torch.complex`
        array of the data values representing :math:`M`
    weight : :class:`torch.Tensor` of :class:`torch.double`
        array of weight values representing :math:`w_i`

    Returns
    -------
    :class:`torch.Tensor` of :class:`torch.double`
        the average of the negative log likelihood, summed over all dimensions of
        input array.
    """
    N = len(torch.ravel(data_vis))  # number of complex visibilities
    ll = log_likelihood(model_vis, data_vis, weight)
    # factor of 2 is because of complex calculation
    return -ll / (2 * N)


def chi_squared_gridded(
    modelVisibilityCube: torch.Tensor, griddedDataset: GriddedDataset
) -> torch.Tensor:
    r"""
    Calculate the :math:`\chi^2` (corresponding to :func:`~mpol.losses.chi_squared`)
    using gridded data and model visibilities.

    Parameters
    ----------
    modelVisibilityCube : :class:`torch.Tensor` of :class:`torch.complex`
        torch tensor with shape ``(nchan, npix, npix)`` to be indexed by the
        ``mask`` from :class:`~mpol.datasets.GriddedDataset`. Assumes tensor is
        "pre-packed," as in output from :meth:`mpol.fourier.FourierCube.forward()`.
    griddedDataset: :class:`~mpol.datasets.GriddedDataset` object
        the gridded dataset, most likely produced from
        :meth:`mpol.gridding.DataAverager.to_pytorch_dataset`

    Returns
    -------
    :class:`torch.Tensor` of :class:`torch.double`
        the :math:`\chi^2` value, summed over all dimensions of input data.
    """

    # get the model_visibilities from the dataset
    # 1D torch tensor collapsed across cube dimensions, like
    # griddedDataset.vis_indexed and griddedDataset.weight_indexed

    model_vis = griddedDataset(modelVisibilityCube)

    return chi_squared(
        model_vis, griddedDataset.vis_indexed, griddedDataset.weight_indexed
    )


def log_likelihood_gridded(
    modelVisibilityCube: torch.Tensor, griddedDataset: GriddedDataset
) -> torch.Tensor:
    r"""

    Calculate :math:`\ln\mathcal{L}` (corresponding to
    :func:`~mpol.losses.log_likelihood`) using gridded quantities.

    Parameters
    ----------
    modelVisibilityCube : :class:`torch.Tensor` of :class:`torch.complex`
        torch tensor with shape ``(nchan, npix, npix)`` to be indexed by the
        ``mask`` from :class:`~mpol.datasets.GriddedDataset`. Assumes tensor is
        "pre-packed," as in output from :meth:`mpol.fourier.FourierCube.forward()`.
    griddedDataset: :class:`~mpol.datasets.GriddedDataset` object
        the gridded dataset, most likely produced from
        :meth:`mpol.gridding.DataAverager.to_pytorch_dataset`

    Returns
    -------
    :class:`torch.Tensor` of :class:`torch.double`
        the :math:`\ln\mathcal{L}` value, summed over all dimensions of input data.
    """

    # get the model_visibilities from the dataset
    # 1D torch tensor collapsed across cube dimensions, like
    # griddedDataset.vis_indexed and griddedDataset.weight_indexed
    model_vis = griddedDataset(modelVisibilityCube)

    return log_likelihood(
        model_vis, griddedDataset.vis_indexed, griddedDataset.weight_indexed
    )


def reduced_chi_squared_gridded(
    modelVisibilityCube: torch.Tensor, griddedDataset: GriddedDataset
) -> torch.Tensor:
    r"""

    Calculate the reduced :math:`\chi^2_\mathrm{r}` between the complex data
    :math:`\boldsymbol{V}` and model :math:`M` visibilities using gridded quantities.
    Function will return the same value regardless of whether Hermitian pairs are
    included.

    Parameters
    ----------
    modelVisibilityCube : :class:`torch.Tensor` of :class:`torch.complex`
        torch tensor with shape ``(nchan, npix, npix)`` to be indexed by the
        ``mask`` from :class:`~mpol.datasets.GriddedDataset`. Assumes tensor is
        "pre-packed," as in output from :meth:`mpol.fourier.FourierCube.forward()`.
    griddedDataset: :class:`~mpol.datasets.GriddedDataset` object
        the gridded dataset, most likely produced from
        :meth:`mpol.gridding.DataAverager.to_pytorch_dataset`

    Returns
    -------
    :class:`torch.Tensor` of :class:`torch.double`
        the :math:`\chi^2_\mathrm{r}` value summed over all input dimensions
    """
    model_vis = griddedDataset(modelVisibilityCube)

    return reduced_chi_squared(
        model_vis, griddedDataset.vis_indexed, griddedDataset.weight_indexed
    )


def entropy(
    cube: torch.Tensor, prior_intensity: torch.Tensor, tot_flux: float = 10
) -> torch.Tensor:
    r"""
    Calculate the entropy loss of a set of pixels following the definition in
    `EHT-IV 2019 <https://ui.adsabs.harvard.edu/abs/2019ApJ...875L...4E/abstract>`_.

    .. math::

        L = \frac{1}{\zeta} \sum_i I_i \; \ln \frac{I_i}{p_i}

    Parameters
    ----------
    cube : :class:`torch.Tensor` of :class:`torch.double`
        pixel values must be positive :math:`I_i > 0` for all :math:`i`
    prior_intensity : :class:`torch.Tensor` of :class:`torch.double`
        the prior value :math:`p` to calculate entropy against. Tensors of any shape
        are allowed so long as they will broadcast to the shape of the cube under
        division (`/`).
    tot_flux : float
        a fixed normalization factor; the user-defined target total flux density, in
        units of Jy.

    Returns
    -------
    :class:`torch.Tensor` of :class:`torch.double`
        entropy loss
    """
    # check to make sure image is positive, otherwise raise an error
    assert (cube >= 0.0).all(), "image cube contained negative pixel values"
    assert prior_intensity > 0, "image prior intensity must be positive"
    assert tot_flux > 0, "target total flux must be positive"

    return (1 / tot_flux) * torch.sum(cube * torch.log(cube / prior_intensity))


def TV_image(sky_cube: torch.Tensor, epsilon: float = 1e-10) -> torch.Tensor:
    r"""
    Calculate the total variation (TV) loss in the image dimension (R.A. and DEC).
    Following the definition in `EHT-IV 2019
    <https://ui.adsabs.harvard.edu/abs/2019ApJ...875L...4E/abstract>`_ Promotes the
    image to be piecewise smooth and the gradient of the image to be sparse.

    .. math::

        L = \sum_{l,m,v} \sqrt{(I_{l + 1, m, v} - I_{l,m,v})^2 +
            (I_{l, m+1, v} - I_{l, m, v})^2 + \epsilon}


    Parameters
    ----------
    sky_cube: 3D :class:`torch.Tensor` of :class:`torch.double`
        the image cube array :math:`I_{lmv}`, where :math:`l`
        is R.A. in :math:`ndim=3`, :math:`m` is DEC in :math:`ndim=2`, and
        :math:`v` is the channel (velocity or frequency) dimension in
        :math:`ndim=1`. Should be in sky format representation.
    epsilon : float
        a softening parameter in units of [:math:`\mathrm{Jy}/\mathrm{arcsec}^2`].
        Any pixel-to-pixel variations within each image North-South or East-West
        slice greater than this parameter will incur a significant penalty.

    Returns
    -------
    :class:`torch.Tensor` of :class:`torch.double`
        total variation loss
    """

    # diff the cube in ll and remove the last row
    diff_ll = sky_cube[:, 0:-1, 1:] - sky_cube[:, 0:-1, 0:-1]

    # diff the cube in mm and remove the last column
    diff_mm = sky_cube[:, 1:, 0:-1] - sky_cube[:, 0:-1, 0:-1]

    loss = torch.sum(torch.sqrt(diff_ll**2 + diff_mm**2 + epsilon))

    return loss


def TV_channel(cube: torch.Tensor, epsilon: float = 1e-10) -> torch.Tensor:
    r"""
    Calculate the total variation (TV) loss in the channel (first) dimension.
    Following the definition in `EHT-IV 2019
    <https://ui.adsabs.harvard.edu/abs/2019ApJ...875L...4E/abstract>`_, calculate

    .. math::

        L = \sum_{l,m,v} \sqrt{(I_{l, m, v + 1} - I_{l,m,v})^2 + \epsilon}

    Parameters
    ----------
    cube: :class:`torch.Tensor` of :class:`torch.double`
        the image cube array :math:`I_{lmv}`
    epsilon: float
        a softening parameter in units of [:math:`\mathrm{Jy}/\mathrm{arcsec}^2`].
        Any channel-to-channel pixel variations greater than this parameter will incur
        a significant penalty.

    Returns
    -------
    :class:`torch.Tensor` of :class:`torch.double`
        total variation loss
    """
    # calculate the difference between the n+1 cube and the n cube
    diff_vel = cube[1:] - cube[0:-1]
    loss = torch.sum(torch.sqrt(diff_vel**2 + epsilon))

    return loss


def TSV(sky_cube: torch.Tensor) -> torch.Tensor:
    r"""
    Calculate the total square variation (TSV) loss in the image dimension
    (R.A. and DEC). Following the definition in `EHT-IV 2019
    <https://ui.adsabs.harvard.edu/abs/2019ApJ...875L...4E/abstract>`_ Promotes the
    image to be edge smoothed which may be a better reoresentation of the truth image
    `K. Kuramochi et al 2018
    <https://ui.adsabs.harvard.edu/abs/2018ApJ...858...56K/abstract>`_.

    .. math::

        L = \sum_{l,m,v} (I_{l + 1, m, v} - I_{l,m,v})^2 +
        (I_{l, m+1, v} - I_{l, m, v})^2

    Parameters
    ----------
    sky_cube :class:`torch.Tensor` of :class:`torch.double`
        the image cube array :math:`I_{lmv}`, where :math:`l`
        is R.A. in :math:`ndim=3`, :math:`m` is DEC in :math:`ndim=2`, and
        :math:`v` is the channel (velocity or frequency) dimension in
        :math:`ndim=1`. Should be in sky format representation.

    Returns
    -------
    :class:`torch.Tensor` of :class:`torch.double`
        total square variation loss

    """

    # diff the cube in ll and remove the last row
    diff_ll = sky_cube[:, 0:-1, 1:] - sky_cube[:, 0:-1, 0:-1]

    # diff the cube in mm and remove the last column
    diff_mm = sky_cube[:, 1:, 0:-1] - sky_cube[:, 0:-1, 0:-1]

    loss = torch.sum(diff_ll**2 + diff_mm**2)

    return loss


def sparsity(cube: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    r"""
    Enforce a sparsity prior on the image cube using the :math:`L_1` norm. Optionally
    provide a boolean mask to apply the prior to only the ``True`` locations. For
    example, you might want this mask to be ``True`` for background regions.

    The sparsity loss calculated as

    .. math::

        L = \sum_i | I_i |

    Parameters
    ----------
    cube : :class:`torch.Tensor` of :class:`torch.double`
        the image cube array :math:`I_{lmv}`
    mask : :class:`torch.Tensor` of :class:`torch.bool`
        tensor array the same shape as ``cube``. The sparsity prior
        will be applied to those pixels where the mask is ``True``. Default is
        to apply prior to all pixels.

    Returns
    -------
    :class:`torch.Tensor` of :class:`torch.double`
        sparsity loss calculated where ``mask == True``
    """

    if mask is not None:
        loss = torch.sum(torch.abs(cube.masked_select(mask)))
    else:
        loss = torch.sum(torch.abs(cube))

    return loss


def UV_sparsity(
    vis: torch.Tensor, qs: torch.Tensor, q_max: torch.Tensor
) -> torch.Tensor:
    r"""
    Enforce a sparsity prior for all :math:`q = \sqrt{u^2 + v^2}` points larger than
    :math:`q_\mathrm{max}`.

    Parameters
    ----------
    vis : :class:`torch.Tensor` of :class:`torch.complex128`
        visibility cube of (nchan, npix, npix//2 +1, 2)
    qs : :class:`torch.Tensor` of :class:`torch.float64`
        array corresponding to visibility coordinates. Dimensionality of
        (npix, npix//2)
    q_max : float
        maximum radial baseline

    Returns
    -------
    :class:`torch.Tensor` of :class:`torch.double`
        UV sparsity loss above :math:`q_\mathrm{max}`
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


def PSD(qs: torch.Tensor, psd: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
    r"""
    Apply a loss function corresponding to the power spectral density using a Gaussian
    process kernel.

    Assumes an image plane kernel of

    .. math::

        k(r) = \exp(-\frac{r^2}{2 \ell^2})

    The corresponding power spectral density is

    .. math::

        P(q) = (2 \pi \ell^2) \exp(- 2 \pi^2 \ell^2 q^2)


    Parameters
    ----------
    qs : :class:`torch.Tensor` of :class:`torch.double`
        the radial UV coordinate (in kilolambda)
    psd : :class:`torch.Tensor` of :class:`torch.double`
        the power spectral density cube
    l : :class:`torch.Tensor` of :class:`torch.double`
        the correlation length in the image plane (in arcsec)

    Returns
    -------
    :class:`torch.Tensor` of :class:`torch.double`
        the loss calculated using the power spectral density

    """

    # stack to the full 3D shape
    qs = qs * 1e3  # lambda

    l_rad = l * constants.arcsec  # radians

    # calculate the expected power spectral density
    expected_PSD = (
        2 * np.pi * l_rad**2 * torch.exp(-2 * np.pi**2 * l_rad**2 * qs**2)
    )

    # evaluate the chi^2 for the PSD, making sure it broadcasts across all channels
    loss = torch.sum(psd / expected_PSD)

    return loss


def edge_clamp(cube: torch.Tensor) -> torch.Tensor:
    r"""
    Promote all pixels at the edge of the image to be zero using an :math:`L_2` norm.

    Parameters
    ----------
    cube: :class:`torch.Tensor` of :class:`torch.double`
        the image cube array :math:`I_{lmv}`

    Returns
    -------
    :class:`torch.Tensor` of :class:`torch.double`
        edge loss
    """

    # find edge pixels
    # all channels
    # pixel edges
    bt_edges = cube[:, (0, -1)]
    lr_edges = cube[:, :, (0, -1)]

    loss = torch.sum(bt_edges**2) + torch.sum(lr_edges**2)

    return loss

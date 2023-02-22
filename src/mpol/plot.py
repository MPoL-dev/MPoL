import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mco 

from astropy.visualization.mpl_normalize import simple_norm

from mpol.utils import loglinspace, torch2npy, packed_cube_to_sky_cube

def get_image_cmap_norm(image, stretch='power', gamma=1.0, asinh_a=0.02):
    """
    Get a colormap normalization to apply to an image. 

    image : array
        An image array.
    stretch : string, default = 'power'
        Transformation to apply to the colormap. 'power' is a
        power law stretch; 'asinh' is an arcsinh stretch.
    gamma : float, default = 1.0
        Index of power law normalization (see matplotlib.colors.PowerNorm).
        gamma=1.0 yields a linear colormap.
    asinh_a : float, default = 0.02
        Scale parameter for an asinh stretch.
    """
    vmax = image.max()

    if stretch == 'power':
        vmin = 0
        norm = mco.PowerNorm(gamma, vmin, vmax)    
    
    elif stretch == 'asinh':
        vmin = max(0, image.min())
        norm = simple_norm(image, stretch='asinh', asinh_a=asinh_a, 
                        min_cut=vmin, max_cut=vmax)

    else:
        raise ValueError("'stretch' must be one of 'asinh' or 'power'.")
    
    return norm


def vis_histogram_fig(dataset, bin_quantity='count', bin_label=None, q_edges=None, 
    phi_edges=None, q_edges1d=None, show_datapoints=False, save_prefix=None):
    r"""
    Generate a figure with 1d and 2d histograms of (u,v)-plane coverage. 
    Histograms can show different data; see `bin_quantity` parameter.

    Parameters
    ----------
    dataset : `mpol.datasets.GriddedDataset` object
    bin_quantity : str or numpy.ndarray, default='count'
        Which quantity to bin:
            - 'count' bins (u,v) points by their count
            - 'weight' bins points by the data weight (inherited from `dataset`)
            - 'vis_real' bins points by data Re(V)
            - 'vis_imag' bins points by data Im(V)
            - A user-supplied numpy.ndarray to be used as 'weights' in np.histogram 
    bin_label : str, default=None
        Label for 1d histogram y-axis and 2d histogram colorbar.
    q_edges : array, optional (default=None), unit=:math:[`k\lambda`] 
        Radial bin edges for the 1d and 2d histogram. If `None`, defaults to 
        12 log-linearly radial bins over [0, 1.1 * maximum baseline in 
        `dataset`].
    phi_edges : array, optional (default=None), unit=[rad] 
        Azimuthal bin edges for the 2d histogram. If `None`, defaults to 
        16 bins over [-\pi, \pi]
    q_edges1d : array, optional (default=None), unit=:math:[`k\lambda`]
        Radial bin edges for a second 1d histogram. If `None`, defaults to 
        50 bins equispaced over [0, 1.1 * maximum baseline in `dataset`].
    show_datapoints : bool, default = False 
        Whether to overplot the raw visibilities in `dataset` on the 2d 
        histogram.
    save_prefix : string, default = None
        Prefix for saved figure name. If None, the figure won't be saved

    Returns
    -------
    fig : Matplotlib `.Figure` instance
        The generated figure
    axes : Matplotlib `~.axes.Axes` class
        Axes of the generated figure

    Notes
    -----
    No assumption or correction is made concerning whether the (u,v) distances 
    are projected or deprojected.
    """

    # convert dataset pytorch tensors to numpy for convenience
    mask_npy = torch2npy(dataset.mask)
    vis_npy = torch2npy(dataset.vis_indexed)
    weight_npy = torch2npy(dataset.weight_indexed)

    # 2D mask for any UV cells that contain visibilities
    # in *any* channel
    stacked_mask = np.any(mask_npy, axis=0)

    # get qs, phis from dataset and turn into 1D lists
    qs = dataset.coords.packed_q_centers_2D[stacked_mask]
    phis = dataset.coords.packed_phi_centers_2D[stacked_mask]

    if isinstance(bin_quantity, np.ndarray): 
        weights = bin_quantity
        hist_lab = bin_label 

    elif bin_quantity == 'count':
        weights = None
        hist_lab = 'Count'
        
    elif bin_quantity == 'weight':
        weights = np.copy(weight_npy)
        hist_lab = 'Weight'

    elif bin_quantity == 'vis_real':
        weights = np.abs(np.real(vis_npy))
        hist_lab = '|Re(V)|'

    elif bin_quantity == 'vis_imag':
        weights = np.abs(np.imag(vis_npy))
        hist_lab = '|Im(V)|'

    else:
        supported_q = ['count', 'weight', 'vis_real', 'vis_imag']
        raise ValueError("`bin_quantity` ({}) must be one of " 
                        "{}, or a user-provided numpy "
                        " array".format(bin_quantity, supported_q))


    # buffer to include longest baselines in last bin
    pad_factor = 1.1 

    if q_edges1d is None:
        # 1d histogram with uniform bins
        q_edges1d = np.arange(0, qs.max() * pad_factor, 50)

    bin_lab = None
    if all(np.diff(q_edges1d)==np.diff(q_edges1d)[0]):
        bin_lab = r'Bin size {:.0f} k$\lambda$'.format(np.diff(q_edges1d)[0])

    # 2d histogram bins
    if q_edges is None:
        q_edges = loglinspace(0, qs.max() * pad_factor, N_log=8, M_linear=5)
    if phi_edges is None:
        phi_edges = np.linspace(-np.pi, np.pi, num=16 + 1)

    H2d, _, _ = np.histogram2d(qs, phis, weights=weights, 
                                bins=[q_edges, phi_edges])


    fig = plt.figure(figsize=(14,6), tight_layout=True)
    
    # 1d histogram with polar plot bins
    ax0 = fig.add_subplot(221)
    ax0.hist(qs, q_edges, weights=weights, fc='#A4A4A4', ec=(0,0,0,0.3), 
            label='Polar plot bins')
    ax0.legend()
    ax0.set_ylabel(hist_lab)
    
    # 1d histogram with (by default) uniform bins
    ax1 = fig.add_subplot(223, sharex=ax0)
    ax1.hist(qs, q_edges1d, weights=weights, fc='#A93226', label=bin_lab)
    if bin_lab:
        ax1.legend()
    ax1.set_ylabel(hist_lab)
    ax1.set_xlabel(r'Baseline [k$\lambda$]')

    # 2d polar histogram
    ax2 = fig.add_subplot(122, polar=True)
    ax2.set_theta_offset(np.pi / 2)

    # discrete colormap
    cmap = plt.cm.get_cmap("plasma")
    discrete_colors = cmap(np.linspace(0, 1, 10))
    cmap = mco.LinearSegmentedColormap.from_list(None, discrete_colors, 10)

    # choose sensible minimum for colormap
    vmin = max(H2d.flatten()[H2d.flatten() > 0].min(), 1)
    norm = mco.LogNorm(vmin=vmin)

    im = ax2.pcolormesh(
        phi_edges, 
        q_edges,
        H2d,
        shading="flat",
        norm=norm,
        cmap=cmap,
        ec=(0,0,0,0.3),
        lw=0.3,
    )

    cbar = plt.colorbar(im, ax=ax2, shrink=1.0)
    cbar.set_label(hist_lab)

    ax2.set_ylim(top=qs.max() * pad_factor)

    if show_datapoints:
        # plot raw visibilities
        ax2.scatter(phis, qs, s=1.5, rasterized=True, linewidths=0.0, c="k", 
                    alpha=0.3)

    if save_prefix is not None:
        fig.savefig(save_prefix + '_vis_histogram.png', dpi=300)
        plt.close()

    return fig, (ax0, ax1, ax2)


def split_diagnostics_fig(splitter, channel=0, save_prefix=None):
    r"""
    Generate a figure showing (u,v) coverage in train and test sets split from 
    a parent dataset.

    Parameters
    ----------
    splitter : `mpol.crossval.RandomCellSplitGridded` object
        Iterator that returns a `(train, test)` pair of `GriddedDataset`s 
        for each iteration.
    channel : int, default=0
        Channel (of the datasets in `splitter`) to use to generate figure
    save_prefix : string, default = None
        Prefix for saved figure name. If None, the figure won't be saved

    Returns
    -------
    fig : Matplotlib `.Figure` instance
        The generated figure
    axes : Matplotlib `~.axes.Axes` class
        Axes of the generated figure

    Notes
    -----
    No assumption or correction is made concerning whether the (u,v) distances 
    are projected or deprojected.
    """
    fig, axes = plt.subplots(nrows=splitter.k, ncols=2, figsize=(4, splitter.k * 2))

    for ii, (train, test) in enumerate(splitter): 
        train_mask = torch2npy(train.ground_mask[channel])
        test_mask = torch2npy(test.ground_mask[channel])
        vis_ext = train.coords.vis_ext

        axes[ii, 0].imshow(train_mask, origin="lower", extent=vis_ext, 
            cmap="Greys", interpolation="none")        
        axes[ii, 1].imshow(test_mask, origin="lower", extent=vis_ext, 
            cmap="Greys", interpolation="none")            

        axes[ii, 0].set_ylabel("k-fold {:}".format(ii))

    axes[0, 0].set_title("Training set ")
    axes[0, 1].set_title("Test set")

    for aa in axes.flatten():
        aa.xaxis.set_ticklabels([])
        aa.yaxis.set_ticklabels([])

    fig.subplots_adjust(left=0.1, hspace=0.0, wspace=0.2)

    if save_prefix is not None:
        fig.savefig(save_prefix + '_split_diag.png', dpi=300)
        plt.close()

    return fig, axes


def train_diagnostics_fig(model, vis_resid, imager, briggs_robust=0.5,
                         losses=[], train_state=None, channel=0, 
                        save_prefix=None):
    """
    Figure for model diagnostics during an optimization loop. For a `model` in 
    a given state, plots the current: 
        - model image (both linear and arcsinh colormap normalization)
        - gradient image
        - Fourier plane residuals
        - dirty image of Fourier plane residuals
        - loss function

    Parameters
    ----------
    model : `torch.nn.Module` object
        A neural network; instance of the `mpol.precomposed.SimpleNet` class.
    vis_resid : array of complex 
        Model loose residual visibility amplitudes of the form (Re(V) + 1j * Im(V))
    imager : `mpol.gridding.DirtyImager` instance 
        Dirty imager used to image Fourier plane residuals
    briggs_robust : float, default=0.5
        Briggs robust value for the dirty image of the Fourier plane residuals        
    losses : list
        Loss value at each epoch in the training loop
    train_state : dict, default=None
        Dictionary containing current training parameter values       
    channel : int, default=0
        Channel (of the datasets in `splitter`) to use to generate figure        
    save_prefix : string, default = None
        Prefix for saved figure name. If None, the figure won't be saved

    Returns
    -------
    fig : Matplotlib `.Figure` instance
        The generated figure
    axes : Matplotlib `~.axes.Axes` class
        Axes of the generated figure
    """
    fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(12, 10))

    fig.suptitle(fig_title)

    mod_im = torch2npy(model.icube.sky_cube[channel])
    mod_grad = torch2npy(packed_cube_to_sky_cube(model.bcube.base_cube.grad)[channel])

    # current model image (linear colormap)
    im = axes[0, 0].imshow(
        mod_im,
        origin="lower",
        interpolation="none",
        extent=model.icube.coords.img_ext,
        cmap="inferno",
        norm=get_image_cmap_norm(mod_im)
    )
    cbar = plt.colorbar(im, ax=axes[0, 0])
    cbar.set_label('Jy arcsec$^{-2}$')
    axes[0,0].set_title("Model image")

    # current model image (asinh colormap)
    im = axes[0, 1].imshow(
        mod_im,
        origin="lower",
        interpolation="none",
        extent=model.icube.coords.img_ext,
        cmap="inferno",
        norm=get_image_cmap_norm(mod_im, stretch='asinh')
    )
    cbar = plt.colorbar(im, ax=axes[0, 1])
    cbar.set_label('Jy arcsec$^{-2}$')
    axes[0,1].set_title("Model image (asinh stretch)")

    # current gradient image
    im = axes[0, 2].imshow(
        mod_grad,
        origin="lower",
        interpolation="none",
        extent=model.icube.coords.img_ext,
        cmap="inferno",
        norm=get_image_cmap_norm(mod_grad)
    )
    cbar = plt.colorbar(im, ax=axes[0, 2])
    cbar.set_label('Jy arcsec$^{-2}$')
    axes[0,2].set_title("Gradient image")


    cbar.set_label('Jy arcsec$^{-2}$')

    if save_prefix is not None:
        fig.savefig(save_prefix + '_train_diagnostics_fig.png', dpi=300)
        plt.close()

    return fig, axes
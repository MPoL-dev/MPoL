import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mco 
from matplotlib.patches import Ellipse
import torch

from astropy.visualization.mpl_normalize import simple_norm

from mpol.fourier import get_vis_residuals
from mpol.gridding import DirtyImager
from mpol.onedim import radialI, radialV
from mpol.utils import loglinspace, torch2npy, packed_cube_to_sky_cube
from mpol.input_output import ProcessFitsImage

def get_image_cmap_norm(image, stretch='power', gamma=1.0, asinh_a=0.02, symmetric=False):
    """
    Get a colormap normalization to apply to an image. 

    image : array
        2D image array.
    stretch : string, default = 'power'
        Transformation to apply to the colormap. 'power' is a
        power law stretch; 'asinh' is an arcsinh stretch.
    gamma : float, default = 1.0
        Index of power law normalization (see matplotlib.colors.PowerNorm).
        gamma=1.0 yields a linear colormap.
    asinh_a : float, default = 0.02
        Scale parameter for an asinh stretch.
    symmetric : bool, default=False 
        Whether the colormap is symmetric about 0
    """
    if symmetric is True:
        vmax = max(abs(image.min()), image.max())
        vmin = -vmax

    else:
        vmax, vmin = image.max(), image.min()
        if stretch == 'power':
            vmin = 0

    if stretch == 'power':
        norm = mco.PowerNorm(gamma, vmin, vmax)    
    
    elif stretch == 'asinh':
        norm = simple_norm(image, stretch='asinh', asinh_a=asinh_a, 
                        min_cut=vmin, max_cut=vmax)

    else:
        raise ValueError("'stretch' must be one of 'asinh' or 'power'.")
    
    return norm


def get_residual_image(model, u, v, V, weights, robust=0.5):
    """ 
    Get a dirty image and colormap normalization for residual visibilities,
    the difference of observed visibilities and an MPoL model sampled at the 
    observed (u,v) points.

    Parameters
    ----------
    model : `torch.nn.Module` object
        Instance of the `mpol.precomposed.SimpleNet` class. Contains model
        visibilities.
    u, v : array, unit=[k\lambda]
        Data u- and v-coordinates
    V : array, unit=[Jy]
        Data visibility amplitudes
    weights : array, unit=[Jy^-2]
        Data weights
    robust : float, default=0.5
        Robust weighting parameter used to create the dirty image of the 
        residual visibilities

    Returns
    -------
    im_resid : 2D image array
        The residual image
    norm_resid : Matplotlib colormap normalization
        Symmetric, linear colormap for `im_resid`
    """
    vis_resid = get_vis_residuals(model, u, v, V)

    resid_imager = DirtyImager(
        coords=model.coords,
        uu=u,
        vv=v,
        weight=weights,
        data_re=np.real(vis_resid),
        data_im=np.imag(vis_resid),
    )
    im_resid, _ = resid_imager.get_dirty_image(weighting="briggs", 
                                               robust=robust, 
                                               unit='Jy/arcsec^2'
                                               )
    # `get_vis_residuals` has already selected a single channel
    im_resid = np.squeeze(im_resid)
    
    norm_resid = get_image_cmap_norm(im_resid, 
                                     stretch='power', 
                                     gamma=1, 
                                     symmetric=True
                                     )

    return im_resid, norm_resid


def plot_image(image, extent, cmap="inferno", norm=None, ax=None, 
               clab=r"I [Jy arcsec$^{-2}$]",
               xlab=r"$\Delta \alpha \cos \delta$ [${}^{\prime\prime}$]",
               ylab=r"$\Delta \delta$ [${}^{\prime\prime}$]",
               ):
    r""" 
    Wrapper for plt.imshow, with colorbar and colormap normalization.

    Parameters
    ----------
    image : array
        2D image array.
    extent : list, len=4
        x- and y-extents of image: [x-min, x-max, y-min, y-max] (see plt.imshow)
    cmap : str, default="inferno
        Matplotlib colormap.
    norm : Matplotlib colormap normalization, default=None
        Image colormap norm. If None, a linear normalization is generated with
        mpol.plot.get_image_cmap_norm
    ax : Matplotlib axis instance, default=None
        Axis on which to plot the image. If None, a new figure is created.
    clab : str, default=r"Jy arcsec$^{-2}$"
        Colorbar axis label
    xlab : str, default="RA offset [arcsec]"
        Image x-axis label.
    ylab : str, default="Dec offset [arcsec]"
        Image y-axis label.

    Returns
    -------
    im : Matplotlib imshow instance
        The plotted image.
    cbar : Matplotlib colorbar instance
        Colorbar for the image.
    """
    if norm is None:
        norm = get_image_cmap_norm(image)

    if ax is None:
        _, ax = plt.subplots()

    im = ax.imshow(
        image,
        origin="lower",
        interpolation="none",
        extent=extent,
        cmap=cmap,
        norm=norm,
    )

    cbar = plt.colorbar(im, ax=ax, location="right", pad=0.1, shrink=0.7)
    cbar.set_label(clab)

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    
    return im, cbar


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
        Iterator that returns a `(train, test)` pair of `GriddedDataset` 
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
    fig, axes = plt.subplots(nrows=2, ncols=splitter.k, figsize=(10,3))

    cmap_train = mco.ListedColormap(['none', 'black'])
    cmap_test = mco.ListedColormap(['none', 'red'])
    
    kw = {"fontsize":8}
    image_kw = {"origin":"lower", "interpolation":"none"}

    fig.suptitle('Training data: black, test data: red', **kw)

    for ii, (train, test) in enumerate(splitter):
        train_mask = torch2npy(train.ground_mask[channel])
        test_mask = torch2npy(test.ground_mask[channel])
        vis_ext = np.array(train.coords.vis_ext) / 1e3

        axes[0, ii].imshow(train_mask, extent=vis_ext, cmap=cmap_train, **image_kw)
        axes[0, ii].imshow(test_mask, extent=vis_ext, cmap=cmap_test, **image_kw)
        axes[1, ii].imshow(test_mask, extent=vis_ext, cmap=cmap_test, **image_kw)

        axes[0, ii].set_title(f"k-fold {ii}", **kw)

    for aa in axes.flatten()[:-1]:
        aa.yaxis.set_ticks_position("both")
        aa.xaxis.set_ticklabels([])
        aa.yaxis.set_ticklabels([])

    ax = axes[1,-1]
    ax.set_xlabel(r'u [M$\lambda$]', **kw)
    ax.set_ylabel(r'v [M$\lambda$]', **kw)
    ax.yaxis.set_ticks_position("both")
    ax.yaxis.tick_right()    
    ax.yaxis.set_label_position("right") 

    fig.subplots_adjust(hspace=0.02, wspace=0, left=0.03, right=0.92, top=0.9, bottom=0.1)

    if save_prefix is not None:
        fig.savefig(save_prefix + '_split_diag.png', dpi=300)
    
    plt.close()

    return fig, axes


def train_diagnostics_fig(model, losses=None, learn_rates=None, fluxes=None, 
                          old_model_image=None, old_model_epoch=None,
                          kfold=None, epoch=None,
                          channel=0, save_prefix=None):
    """
    Figure for model diagnostics at a given model state during an optimization loop. 
    
    Plots:
        - model image
        - flux of model image
        - gradient image
        - difference image between `old_model_image` and current model image
        - loss function
        - learning rate

    Parameters
    ----------
    model : `torch.nn.Module` object
        A neural network module; instance of the `mpol.precomposed.SimpleNet` class.
    losses : list
        Loss value at each epoch in the training loop
    learn_rates : list
        Learning rate at each epoch in the training loop  
    fluxes : list
        Total flux in model image at each epoch in the training loop
    old_model_image : 2D image array, default=None
        Model image of a previous epoch for comparison to current image  
    old_model_epoch : int
        Epoch of `old_model_image`
    kfold : int, default=None
        Current cross-validation k-fold
    epoch : int, default=None
        Current training epoch
    channel : int, default=0
        Channel (of the datasets in `splitter`) to use to generate figure        
    save_prefix : str, default = None
        Prefix for saved figure name. If None, the figure won't be saved

    Returns
    -------
    fig : Matplotlib `.Figure` instance
        The generated figure
    axes : Matplotlib `~.axes.Axes` class
        Axes of the generated figure
    """
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8, 8))
    axes[1][1].remove()

    fig.suptitle(f"Pixel size {model.coords.cell_size * 1e3:.2f} mas, N_pix {model.coords.npix}\nk-fold {kfold}, epoch {epoch}", fontsize=10)
    
    mod_im = torch2npy(model.icube.sky_cube[channel])
    mod_grad = torch2npy(packed_cube_to_sky_cube(model.bcube.base_cube.grad)[channel])
    extent = model.icube.coords.img_ext

    # model image (linear colormap)
    # ax = axes[0,0]
    # plot_image(mod_im, extent, ax=ax, xlab='', ylab='')
    # ax.set_title("Model image")

    # model image (asinh colormap)
    ax = axes[0,0]
    plot_image(mod_im, extent, ax=ax, xlab='', ylab='', norm=get_image_cmap_norm(mod_im, stretch='asinh'))
    ax.set_title("Model image", fontsize=10)

    # gradient image
    ax = axes[1,0]
    plot_image(mod_grad, extent, ax=ax)
    ax.set_title("Gradient image", fontsize=10)

    if old_model_image is not None:
        # current model image - model image at last stored epoch
        ax = axes[0,1]
        diff_image = mod_im - old_model_image
        diff_im_norm = get_image_cmap_norm(diff_image, symmetric=True)
        plot_image(diff_image, extent, cmap='RdBu_r', ax=ax, xlab='', ylab='', norm=diff_im_norm)
        ax.set_title(f"Difference (epoch {epoch} - {old_model_epoch})", fontsize=10)
        
    if losses is not None:
        # loss function
        ax = fig.add_subplot(426)
        ax.semilogy(losses, 'k', label=f"{losses[-1]:.3f}")
        ax.legend(loc='upper right')
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.set_ylabel('Loss')

    if learn_rates is not None:    
        # learning rate
        ax = fig.add_subplot(428)
        ax.plot(learn_rates, 'k', label=f"{learn_rates[-1]:.3e}")
        ax.legend(loc='upper right')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learn rate')

    plt.tight_layout()

    if fluxes is not None:
        # total flux in model image 
        ax = fig.add_axes([0.08, 0.465, 0.3, 0.08])
        ax.plot(fluxes, 'k', label=f"{fluxes[-1]:.4f}")
        ax.legend(loc='upper right', fontsize=8)
        ax.tick_params(labelsize=8)
        # ax.set_xlabel('Epoch', fontsize=8)
        ax.set_ylabel('Flux [Jy]', fontsize=8)

    if save_prefix is not None:
        fig.savefig(save_prefix + f"_train_diag_kfold{kfold}_epoch{epoch:05d}.png", dpi=300)
    
    plt.close()

    return fig, axes


def crossval_diagnostics_fig(cv, title="", save_prefix=None):
    """
    Figure for model diagnostics of a cross-validation run. 
    
    Plots: 
        - loss evolution for each k-fold
        - cross-validation score per k-fold

    Parameters
    ----------
    cv : `mpol.crossval.CrossValidate` object
        Instance of the `CrossValidate` class produced by a cross-validation loop
    title : str, default=""
        Figure super-title
    save_prefix : string, default = None
        Prefix for saved figure name. If None, the figure won't be saved

    Returns
    -------
    fig : Matplotlib `.Figure` instance
        The generated figure
    axes : Matplotlib `~.axes.Axes` class
        Axes of the generated figure
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6,3))

    title += f"\nRegularizers {cv.regularizers}\nSplit method: {cv.split_method}, CV score {cv.score['mean']:.3f} +- {cv.score['std']:.3f}"
    fig.suptitle(title, fontsize=6)

    axes[0].plot(cv.score['all'], 'k.')
    axes[0].axhline(y=cv.score['mean'], c='r', ls='--', label=r'$\mu$')
    axes[0].axhline(y=cv.score['mean'] + cv.score['std'], c='c', ls=':', label=r'$\pm 1 \sigma$')
    axes[0].axhline(y=cv.score['mean'] - cv.score['std'], c='c', ls=':')

    for i,l in enumerate(cv.diagnostics['loss_histories']):
        axes[1].loglog(l, label=f"k-fold {i}")
    
    axes[0].legend(fontsize=6)
    axes[0].set_xlabel("k-fold")
    axes[0].set_ylabel("Score")

    axes[1].legend(fontsize=6)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    
    plt.tight_layout()

    if save_prefix is not None:
        fig.savefig(save_prefix + "_crossval_diagnostics.png", dpi=300)
    
    plt.close()

    return fig, axes


def image_comparison_fig(model, u, v, V, weights, robust=0.5, 
                         clean_fits=None, share_cscale=False, 
                         xzoom=[None, None], yzoom=[None, None],
                         title="",
                         channel=0, 
                         save_prefix=None):
    """
    Figure for comparison of MPoL model image to other image models. 
    
    Plots: 
        - dirty image
        - MPoL model image
        - MPoL residual visibilities imaged
        - clean image (if a .fits file is supplied)

    Parameters
    ----------
    model : `torch.nn.Module` object
        A neural network; instance of the `mpol.precomposed.SimpleNet` class.
    u, v : array, unit=[k\lambda]
        Data u- and v-coordinates
    V : array, unit=[Jy]
        Data visibility amplitudes
    weights : array, unit=[Jy^-2]
        Data weights        
    robust : float, default=0.5
        Robust weighting parameter used to create the dirty image of the 
        observed visibilities and separately of the MPoL residual visibilities  
    clean_fits : str, default=None
        Path to a clean .fits image
    share_cscale : bool, default=False
        Whether the MPoL model image, dirty image and clean image share the 
        same colorscale
    xzoom, yzoom : list of float, default = [None, None]
        X- and y- axis limits to zoom the images to. `xzoom` and `yzoom` should 
        both list values in ascending order (e.g. [-2, 3], not [3, -2])
    title : str, default=""
        Figure super-title
    channel : int, default=0
        Channel of the model to use to generate figure        
    save_prefix : string, default = None
        Prefix for saved figure name. If None, the figure won't be saved

    Returns
    -------
    fig : Matplotlib `.Figure` instance
        The generated figure
    axes : Matplotlib `~.axes.Axes` class
        Axes of the generated figure
    """
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,10))

    title += f"\nMPoL pixel size {model.coords.cell_size * 1e3:.2f} mas, N_pix {model.coords.npix}"
    if share_cscale:
        title += "\nDirty and clean images use colorscale of MPoL image"
    fig.suptitle(title)

    # get MPoL model image
    mod_im = torch2npy(model.icube.sky_cube[channel])
    total_flux = model.coords.cell_size ** 2 * np.sum(mod_im)

    # get imaged MPoL residual visibilities
    im_resid, norm_resid = get_residual_image(model, u, v, V, weights, robust=robust)

    # get dirty image
    imager = DirtyImager(
        coords=model.coords,
        uu=u,
        vv=v,
        weight=weights,
        data_re=V.real,
        data_im=V.imag
    )
    dirty_im, dirty_beam = imager.get_dirty_image(weighting="briggs",
                                        robust=robust,
                                        unit="Jy/arcsec^2")
    dirty_im = np.squeeze(dirty_im)

    # get clean image and beam
    if clean_fits is not None:
        fits_obj = ProcessFitsImage(clean_fits)
        clean_im, clean_im_ext, clean_beam = fits_obj.get_image(beam=True)

    # set image colorscales
    norm_mod = get_image_cmap_norm(mod_im, stretch='asinh')
    if share_cscale:
        norm_dirty = norm_clean = norm_mod
    else:
        norm_dirty = get_image_cmap_norm(dirty_im, stretch='asinh')
        if clean_fits is not None:
            norm_clean = get_image_cmap_norm(clean_im, stretch='asinh')
    
    # MPoL model image
    plot_image(mod_im, extent=model.icube.coords.img_ext,
                    ax=axes[0][1], norm=norm_mod, xlab='', ylab='')

    # imaged MPoL residual visibilities
    plot_image(im_resid, extent=model.icube.coords.img_ext, 
                ax=axes[1][1], norm=norm_resid, cmap='RdBu_r', xlab='', ylab='')

    # dirty image
    plot_image(dirty_im, extent=model.icube.coords.img_ext,  
                    ax=axes[0][0], norm=norm_dirty)
    
    # clean image
    if clean_fits is not None:
        plot_image(clean_im, extent=clean_im_ext, 
                    ax=axes[1][0], norm=norm_clean, xlab='', ylab='')
        
        # add clean beam to plot
        if any(xzoom) and any(yzoom):
            beam_xy = (0.85 * xzoom[1], 0.85 * yzoom[0])
        else:
            beam_xy = (0.85 * axes[1][0].get_xlim()[1], 0.85 * axes[1][0].get_ylim()[0])

        beam_ellipse = Ellipse(xy=beam_xy,
                            width=clean_beam[0], 
                            height=clean_beam[1], 
                            angle=-clean_beam[2], 
                            color='w'
                            )
        axes[1][0].add_artist(beam_ellipse)

    if any(xzoom) and any(yzoom):
        for ii in [0,1]:
            for jj in [0,1]:
                axes[ii][jj].set_xlim(xzoom[1], xzoom[0])
                axes[ii][jj].set_ylim(yzoom[0], yzoom[1])

    axes[0][0].set_title(f"Dirty image (robust {robust})")
    axes[0][1].set_title(f"MPoL image (flux {total_flux:.4f} Jy)")
    axes[1][1].set_title(f"MPoL residual V imaged (robust {robust})")      
    if clean_fits is not None:
        axes[1][0].set_title(f"Clean image (beam {clean_beam[0] * 1e3:.0f} $\\times$ {clean_beam[1] * 1e3:.0f} mas)")  

    plt.tight_layout()

    if save_prefix is not None:
        fig.savefig(save_prefix + "_image_comparison.png", dpi=300)

    plt.close()

    return fig, axes


def vis_1d_fig(model, u, v, V, weights, geom=None, rescale_flux=False, 
              bin_width=20e3, q_logx=True, title="", channel=0, save_prefix=None):
    """
    Figure for comparison of 1D projected MPoL model visibilities and observed 
        visibilities. 
    
    Plots:
        - Re(V): observed and MPoL model (projected unless `geom` is supplied)
        - Residual Re(V): observed - MPoL model (projected unless `geom` is supplied)
        - Im(V): observed and MPoL model (projected unless `geom` is supplied)
        - Residual Im(V): observed - MPoL model (projected unless `geom` is supplied)

    Parameters
    ----------
    model : `torch.nn.Module` object
        A neural network; instance of the `mpol.precomposed.SimpleNet` class.
    u, v : array, unit=[k\lambda]
        Data u- and v-coordinates
    V : array, unit=[Jy]
        Data visibility amplitudes
    weights : array, unit=[Jy^-2]
        Data weights        
    geom : dict
        Dictionary of source geometry. If passed in, visibilities will be 
            deprojected prior to plotting. Keys:
                "incl" : float, unit=[deg]
                    Inclination 
                "Omega" : float, unit=[deg]
                    Position angle of the ascending node 
                "omega" : float, unit=[deg]
                    Argument of periastron
                "dRA" : float, unit=[arcsec]
                    Phase center offset in right ascension. Positive is west of north.
                "dDec" : float, unit=[arcsec]
                    Phase center offset in declination.
    rescale_flux : bool
        If True, the visibility amplitudes are rescaled to account 
            for the difference between the inclined (observed) brightness and the 
            assumed face-on brightness, assuming the emission is optically thick. 
            The source's integrated (2D) flux is assumed to be:
            :math:`F = \cos(i) \int_r^{r=R}{I(r) 2 \pi r dr}`.
            No rescaling would be appropriate in the optically thin limit.                 
    bin_width : float, default=20e3
        Bin size [klambda] for baselines
    q_logx : bool, default=True
        Whether to plot visibilities in log-baseline
    title : str, default=""
        Figure super-title
    channel : int, default=0
        Channel of the model to use to generate figure        
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
    This routine requires the `frank <https://github.com/discsim/frank>`_ package
    """
    from frank.geometry import apply_phase_shift, deproject
    from frank.utilities import UVDataBinner

    # get MPoL residual and model visibilities
    Vresid, Vmod = get_vis_residuals(model, u, v, V, return_Vmod=True)

    if geom is not None:    
        # phase-shift the visibilities
        V = apply_phase_shift(u * 1e3, v * 1e3, V, geom["dRA"], geom["dDec"], inverse=True)
        Vmod = apply_phase_shift(u * 1e3, v * 1e3, Vmod, geom["dRA"], geom["dDec"], inverse=True)
        Vresid = apply_phase_shift(u * 1e3, v * 1e3, Vresid, geom["dRA"], geom["dDec"], inverse=True)

        # deproject the (u,v) points
        u, v, _ = deproject(u * 1e3, v * 1e3, geom["incl"], geom["Omega"])
        # convert back to [k\lambda]
        u /= 1e3
        v /= 1e3

        # if the source is optically thick, rescale the deprojected V(q)
        if rescale_flux: 
            V.real /= np.cos(geom["incl"] * np.pi / 180)
            Vmod.real /= np.cos(geom["incl"] * np.pi / 180)
            Vresid.real /= np.cos(geom["incl"] * np.pi / 180)
            weights *= np.cos(geom["incl"] * np.pi / 180) ** 2

    # bin projected observed visibilities
    # (`UVDataBinner` expects `u`, `v` in [lambda])
    binned_Vtrue = UVDataBinner(np.hypot(u * 1e3, v * 1e3), V, weights, bin_width)

    # bin projected model and residual visibilities
    binned_Vmod = UVDataBinner(np.hypot(u * 1e3, v * 1e3), Vmod, weights, bin_width)
    binned_Vresid = UVDataBinner(np.hypot(u * 1e3, v * 1e3), Vresid, weights, bin_width)

    # baselines [Mlambda]
    qq = binned_Vtrue.uv / 1e6

    amax_binVres_re = np.max(abs(binned_Vresid.V.real))
    amax_binVres_im = np.max(abs(binned_Vresid.V.imag))

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10,8))

    if geom is None:
        title += "\nProjected visibilities"
    else:
        title += "\nDeprojected visibilities"
        if rescale_flux:
            title += "\nRe(V) and weights rescaled for optically thick source"
    fig.suptitle(title)

    # *projected* Re(V) -- observed and MPoL model
    axes[0].plot(qq, binned_Vtrue.V.real * 1e3, 'k.', 
                    label=f"Obs., {bin_width / 1e3:.2f} k$\\lambda$ bins")
    axes[0].plot(qq, binned_Vmod.V.real * 1e3, 'r.', 
                    label='MPoL')
    axes[0].legend()

    # *projected* Im(V) -- observed and MPoL model
    axes[2].plot(qq, binned_Vtrue.V.imag * 1e3, 'k.')
    axes[2].plot(qq, binned_Vmod.V.imag * 1e3, 'r.')

    # *projected* residual Re(V) = observed - MPoL model
    axes[1].plot(qq, binned_Vresid.V.real * 1e3, '.', c='#33C1FF',
                    label=f"Mean {np.mean(binned_Vresid.V.real) * 1e3:.1e} mJy")
    axes[1].legend()

    # *projected* residual Im(V) = observed - MPoL model
    axes[3].plot(qq, binned_Vresid.V.imag * 1e3, '.', c='#33C1FF',
                    label=f"Mean {np.mean(binned_Vresid.V.imag) * 1e3:.1e} mJy")
    axes[3].legend()

    # y-lims on residual plots symmetric about 0
    axes[1].set_ylim(-amax_binVres_re * 1e3, amax_binVres_re * 1e3)
    axes[3].set_ylim(-amax_binVres_im * 1e3, amax_binVres_im * 1e3)
    axes[1].axhline(y=0, ls='--', c='k')
    axes[3].axhline(y=0, ls='--', c='k')

    for ii in range(4):
        axes[ii].set_xlim(0.9 * np.min(qq), 1.1 * np.max(qq))
        if q_logx:
            axes[ii].set_xscale('log')
        if ii < 3:
            axes[ii].xaxis.set_tick_params(labelbottom=False)

    axes[0].set_ylabel('Re(V) [mJy]')
    axes[1].set_ylabel('Resid. Re(V) [mJy]')            
    axes[2].set_ylabel('Im(V) [mJy]')
    axes[3].set_ylabel('Resid. Im(V) [mJy]')
    axes[3].set_xlabel(r'Baseline [M$\lambda$]')

    plt.tight_layout()

    if save_prefix is not None:
        if geom is None:
            suffix = "_projected_"
        else:
            suffix = "_deprojected_"
            if rescale_flux is True:
                suffix += "rescaled_"

        fig.savefig(save_prefix + suffix + "visibilities.png", dpi=300)

    plt.close()

    return fig, axes
    

def radial_fig(model, geom, u=None, v=None, V=None, weights=None, dist=None, 
               rescale_flux=False, bin_width=20e3, q_logx=True, title="", 
               channel=0, save_prefix=None):
    """
    Figure for analysis of 1D (radial) brightness profile of MPoL model image,
    using a user-supplied geometry. 
    
    Plots:
        - MPoL model image
        - 1D (radial) brightness profile extracted from MPoL image (supply `dist` to show second x-axis in [AU])
        - Deprojectd Re(V): binned MPoL model and observations (if u, v, V, weights supplied)
        - Deprojected Im(V): binned MPoL model and observations (if u, v, V, weights supplied)

    Parameters
    ----------
    model : `torch.nn.Module` object
        A neural network; instance of the `mpol.precomposed.SimpleNet` class.
    geom : dict
        Dictionary of source geometry. Used to deproject image and visibilities.
            Keys:
                "incl" : float, unit=[deg]
                    Inclination 
                "Omega" : float, unit=[deg]
                    Position angle of the ascending node 
                "omega" : float, unit=[deg]
                    Argument of periastron
                "dRA" : float, unit=[arcsec]
                    Phase center offset in right ascension. Positive is west of north.
                "dDec" : float, unit=[arcsec]
                    Phase center offset in declination.
    u, v : array, optional, unit=[k\lambda], default=None
        Data u- and v-coordinates
    V : array, optional, unit=[Jy], default=None
        Data visibility amplitudes
    weights : array, optional, unit=[Jy^-2], default=None
        Data weights        
    dist : float, optional, unit = [AU], default = None
        Distance to source, used to show second x-axis for I(r) in [AU]                  
    rescale_flux : bool
        If True, the visibility amplitudes are rescaled to account 
        for the difference between the inclined (observed) brightness and the 
        assumed face-on brightness, assuming the emission is optically thick. 
        The source's integrated (2D) flux is assumed to be:
        :math:`F = \cos(i) \int_r^{r=R}{I(r) 2 \pi r dr}`.
        No rescaling would be appropriate in the optically thin limit.                 
    bin_width : float, default=20e3
        Bin size [klambda] in which to bin observed visibility points
    q_logx : bool, default=True
        Whether to plot visibilities in log-baseline
    title : str, default=""
        Figure super-title
    channel : int, default=0
        Channel of the model to use to generate figure        
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
    This routine requires the `frank <https://github.com/discsim/frank>`_ package
    """
    if not any(x is None for x in [u, v, V, weights]):
        from frank.geometry import apply_phase_shift, deproject
        from frank.utilities import UVDataBinner

        # phase-shift the observed visibilities
        V = apply_phase_shift(u * 1e3, v * 1e3, V, geom["dRA"], geom["dDec"], inverse=True)

        # deproject the observed (u,v) points
        u, v, _ = deproject(u * 1e3, v * 1e3, geom["incl"], geom["Omega"])
        # convert back to [k\lambda]
        u /= 1e3
        v /= 1e3

        # if the source is optically thick, rescale the deprojected V(q)
        if rescale_flux: 
            V.real /= np.cos(geom["incl"] * np.pi / 180)
            weights *= np.cos(geom["incl"] * np.pi / 180) ** 2

        # bin observed visibilities
        # (`UVDataBinner` expects `u`, `v` in [lambda])
        binned_Vtrue = UVDataBinner(np.hypot(u * 1e3, v * 1e3), V, weights, bin_width)

    # model radial image profile
    rs, Is = radialI(model.icube, geom)

    # model radial visibility profile
    q_mod, V_mod = radialV(model.fcube, geom, rescale_flux=rescale_flux)

    # MPoL model image
    mod_im = torch2npy(model.icube.sky_cube[channel])
    total_flux = model.coords.cell_size ** 2 * np.sum(mod_im)


    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
    axes = axes.ravel()

    title += f"\nGeometry (units: deg, arcsec):\n{geom}"
    fig.suptitle(title)

    # MPoL model image
    norm_mod = get_image_cmap_norm(mod_im, stretch='asinh')    
    plot_image(mod_im, extent=model.icube.coords.img_ext, ax=axes[0], norm=norm_mod)
    
    # MPoL model I(r_arcsec)
    axes[2].plot(rs, Is, 'r-', label='MPoL')
    axes[2].legend()

    # I(r_AU) 
    if dist is not None:
        ax2top = axes[2].twiny()
        ax2top.plot(rs * dist, Is, 'r-')

    # Re(V) -- observed and MPoL model
    if not any(x is None for x in [u, v, V, weights]):
        axes[1].plot(binned_Vtrue.uv / 1e6, binned_Vtrue.V.real * 1e3, 'k.', 
                    label=f"Obs., {bin_width / 1e3:.2f} k$\\lambda$ bins")
    axes[1].plot(q_mod / 1e3, V_mod.real * 1e3, 'r.-', label='MPoL')
    axes[1].legend()

    # Im(V) -- observed and MPoL model
    if not any(x is None for x in [u, v, V, weights]):
        axes[3].plot(binned_Vtrue.uv / 1e6, binned_Vtrue.V.imag * 1e3, 'k.')
    axes[3].plot(q_mod / 1e3, V_mod.imag * 1e3, 'r.-')

    for ii in [1,3]:
        if q_logx:
            axes[ii].set_xscale('log')
        if not any(x is None for x in [u, v]):
            q_obs = np.hypot(u, v)
            axes[ii].set_xlim(right=1.1 * np.max(q_obs) / 1e3)
        else:
            axes[ii].set_xlim(right=1.1 * np.max(q_mod) / 1e3)

    axes[0].set_title(f"MPoL image (flux {total_flux:.4f} Jy)")
    axes[2].set_ylabel(r'I [Jy / arcsec$^2$]')
    axes[2].set_xlabel(r'r [arcsec]')
    if dist is not None:
        ax2top.spines['top'].set_color('#1A9E46')
        ax2top.tick_params(axis='x', which='both', colors='#1A9E46')
        ax2top.set_xlabel('r [AU]', color='#1A9E46')
        xlims = axes[2].get_xlim()
        ax2top.set_xlim(np.multiply(xlims, dist))
        
    axes[1].xaxis.set_tick_params(labelbottom=False)
    axes[1].set_ylabel('Re(V) [mJy]')
    axes[3].set_ylabel('Im(V) [mJy]')
    axes[3].set_xlabel(r'Baseline [M$\lambda$]')

    plt.tight_layout()

    if save_prefix is not None:
        fig.savefig(save_prefix + "_radial_profiles.png", dpi=300)
    
    plt.close()

    return fig, axes
    
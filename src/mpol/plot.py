import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.colors as mco 

from mpol.utils import loglinspace

def vis_histogram(dataset, show_weights=False, q_edges=None, phi_edges=None, 
    q_edges1d=None, cmap=None, norm=None, show_datapoints=False, 
    save_prefix=None):

    # 2D mask for any UV cells that contain visibilities
    # in *any* channel
    stacked_mask = np.any(dataset.mask.detach().cpu().numpy(), axis=0)

    # get qs, phis from dataset and turn into 1D lists
    qs = dataset.coords.packed_q_centers_2D[stacked_mask]
    phis = dataset.coords.packed_phi_centers_2D[stacked_mask]

    if show_weights:
        # weight histogram members using data weights, 
        # normalized to mean data weight across full dataset
        weights = dataset.weight_indexed.detach().cpu().numpy()
        weights = weights / weights.mean() 
        hist_lab = 'Sensitivity-weighted count,\n' + \
                    r'$c_i = w_i / w_{\rm mean}$'
    else:
        weights = None
        hist_lab = 'Count'

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

    if cmap is None:
        # discrete colormap
        cmap = plt.cm.get_cmap("plasma")
        discrete_colors = cmap(np.linspace(0, 1, 10))
        cmap = mco.LinearSegmentedColormap.from_list(None, discrete_colors, 10)
    if norm is None:
        norm = mco.LogNorm(vmin=1)

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

    if save_prefix:
        fig.savefig(save_prefix + '_vis_histogram.png', dpi=300)

    return fig, (ax0, ax1, ax2)
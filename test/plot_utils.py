import matplotlib.pyplot as plt
import torch
import numpy as np


def imshow_two(path, imgs, **kwargs):
    """Plot two images side by side, with scalebars.
    
    imgs is a list
    Parameters
    ----------
    path : string
        path and filename to save figure
    imgs : list
        length-2 list of images to plot. Arguments are designed to be very permissive. If the image is a PyTorch tensor, the routine converts it to numpy, and then numpy.squeeze is called.
    titles: list
        if provided, list of strings corresponding to title for each subplot.
    
        
    Returns
    -------
    None
    """

    xx = 7.1  # in
    rmargin = 0.8
    lmargin = 0.8
    tmargin = 0.3
    bmargin = 0.5
    middle_sep = 1.2
    ax_width = (xx - rmargin - lmargin - middle_sep) / 2
    ax_height = ax_width
    cax_width = 0.1
    cax_sep = 0.15
    cax_height = ax_height
    yy = bmargin + ax_height + tmargin

    fig = plt.figure(figsize=(xx, yy))

    ax = []
    cax = []
    for i in [0, 1]:
        ax.append(
            fig.add_axes(
                [
                    (lmargin + i * (ax_width + middle_sep)) / xx,
                    bmargin / yy,
                    ax_width / xx,
                    ax_height / yy,
                ]
            )
        )
        cax.append(
            fig.add_axes(
                (
                    [
                        (lmargin + (i + 1) * ax_width + i * middle_sep + cax_sep) / xx,
                        bmargin / yy,
                        cax_width / xx,
                        cax_height / yy,
                    ]
                )
            )
        )

        img = imgs[i]
        img = img.detach().numpy() if torch.is_tensor(img) else img

        im = ax[i].imshow(np.squeeze(img), origin="lower", interpolation="none")
        plt.colorbar(im, cax=cax[i])
        
        if "titles" in kwargs:
            ax[i].set_title(kwargs["titles"][i])


    fig.savefig(path)

import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import numpy as np


def extend_list(l, num=2):
    """
    Duplicate or extend a list to two items.

    l: list
        the list of items to potentially duplicate or truncate.
    num: int
        the final length of the list

    Returns
    -------
    list
        Length num list of items.

    Examples
    --------
    >>> extend_list(["L Plot", "R Plot"])
    ["L Plot", "R Plot"]
    >>> extend_list({["Plot"]) # both L and R will have "Plot"
    ["Plot", "Plot"]
    >>> extend_list({["L Plot", "R Plot", "Z Plot"]}) # "Z Plot" is ignored
    ["L Plot", "R Plot"]
    """
    if len(l) == 1:
        return num * l
    else:
        return l[:num]
    
def extend_kwargs(kwargs):
    """
    This is a helper routine for imshow_two, designed to flexibly consume a variety
    of options for each of the two plots.

    kwargs: dict
        the kwargs dict provided from the function call
    
    Returns
    -------
    dict
        Updated kwargs with length 2 lists of items.
    """
    
    for key, item in kwargs.items():
        kwargs[key] = extend_list(item)

def imshow_two(path, imgs, sky=False, suptitle=None, **kwargs):
    """Plot two images side by side, with scalebars.
    
    imgs is a list
    Parameters
    ----------
    path : string
        path and filename to save figure
    imgs : list
        length-2 list of images to plot. Arguments are designed to be very permissive. If the image is a PyTorch tensor, the routine converts it to numpy, and then numpy.squeeze is called.
    sky: bool
        If True, treat images as sky plots and label with offset arcseconds.
    title: list
        if provided, list of strings corresponding to title for each subplot. If only one provided,
    xlabel: list
        if provided, list of strings
    
        
    Returns
    -------
    None
    """

    xx = 7.5  # in
    rmargin = 0.8
    lmargin = 0.8
    tmargin = 0.3 if suptitle is None else 0.5
    bmargin = 0.5
    middle_sep = 1.3
    ax_width = (xx - rmargin - lmargin - middle_sep) / 2
    ax_height = ax_width
    cax_width = 0.1
    cax_sep = 0.15
    cax_height = ax_height
    yy = bmargin + ax_height + tmargin

    with mpl.rc_context({'figure.autolayout': False}):
        fig = plt.figure(figsize=(xx, yy))

        ax = []
        cax = []
        
        extend_kwargs(kwargs) 

        if "extent" not in kwargs:
            kwargs["extent"] = [None, None]
            
        for i in [0, 1]:
            a = fig.add_axes(
                    [
                        (lmargin + i * (ax_width + middle_sep)) / xx,
                        bmargin / yy,
                        ax_width / xx,
                        ax_height / yy,
                    ]
                )
            ax.append(a)
                
            ca = fig.add_axes(
                (
                    [
                        (lmargin + (i + 1) * ax_width + i * middle_sep + cax_sep) / xx,
                        bmargin / yy,
                        cax_width / xx,
                        cax_height / yy,
                    ]
                )
            )
            cax.append(ca)
                
            img = imgs[i]
            img = img.detach().numpy() if torch.is_tensor(img) else img

            im = a.imshow(np.squeeze(img), origin="lower", interpolation="none", extent=kwargs["extent"][i])
            plt.colorbar(im, cax=ca)
            
            if "title" in kwargs:
                a.set_title(kwargs["title"][i])

            if sky:
                a.set_xlabel(r"$\Delta \alpha\ \cos \delta\;[{}^{\prime\prime}]$")
                a.set_ylabel(r"$\Delta \delta\;[{}^{\prime\prime}]$")
            else:
                if "xlabel" in kwargs:
                    a.set_xlabel(kwargs["xlabel"][i])

                if "ylabel" in kwargs:
                    a.set_ylabel(kwargs["ylabel"][i])

        if suptitle is not None:
            fig.suptitle(suptitle)
        fig.savefig(path)
        plt.close("all")

import torch
import numpy as np
import mpol.utils
import matplotlib.pyplot as plt


def test_mpol_fftshift():

    # create a fake image
    xx, yy = np.mgrid[0:20, 0:20]
    image_init = xx + 2 * yy

    # initialize it as a torch matrix
    image_torch = torch.tensor(image_init)

    # try fftshift to see if we got it
    shifted_torch = mpol.utils.fftshift(image_torch, axes=(1,))
    shifted_numpy = np.fft.fftshift(image_init, axes=1)

    fig, ax = plt.subplots(ncols=3)

    # compare to actual fftshift and diff
    ax[0].imshow(shifted_numpy, origin="upper")
    ax[1].imshow(shifted_torch.detach().numpy(), origin="upper")
    ax[2].imshow(shifted_numpy - shifted_torch.detach().numpy(), origin="upper")
    fig.savefig("test/fftshift.png")

    assert np.allclose(
        shifted_numpy, shifted_torch.detach().numpy()
    ), "fftshifts do not match"

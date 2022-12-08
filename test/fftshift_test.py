import matplotlib.pyplot as plt
import numpy as np
import torch

import mpol.utils


def test_mpol_fftshift(tmp_path):

    # create a fake image
    xx, yy = np.mgrid[0:20, 0:20]
    image_init = xx + 2 * yy

    # initialize it as a torch matrix
    image_torch = torch.tensor(image_init)

    # try torch fftshift to see if we understand it correctly
    shifted_torch = torch.fft.fftshift(image_torch, dim=(1,))
    shifted_numpy = np.fft.fftshift(image_init, axes=1)

    fig, ax = plt.subplots(ncols=3)

    # compare to actual fftshift and diff
    ax[0].imshow(shifted_numpy, origin="upper")
    ax[1].imshow(shifted_torch.detach().numpy(), origin="upper")
    ax[2].imshow(shifted_numpy - shifted_torch.detach().numpy(), origin="upper")
    fig.savefig(str(tmp_path / "fftshift.png"))

    assert np.allclose(
        shifted_numpy, shifted_torch.detach().numpy()
    ), "fftshifts do not match"

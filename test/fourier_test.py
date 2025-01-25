import matplotlib.pyplot as plt
import numpy as np
import torch
from mpol import fourier, utils
from pytest import approx

# parameters for analytic Gaussian used for all tests
gauss_kw = {
        "a": 1,
        "delta_x": 0.02,  # arcsec
        "delta_y": -0.01,
        "sigma_x": 0.02,
        "sigma_y": 0.01,
        "Omega": 20,  # degrees
    }

def test_fourier_cube(coords, tmp_path):
    # test image packing
    # test whether we get the same Fourier Transform using the FFT as we could
    # calculate analytically

    
    img_packed = utils.sky_gaussian_arcsec(
        coords.packed_x_centers_2D, coords.packed_y_centers_2D, **gauss_kw
    )

    # calculated the packed FFT using the FourierLayer
    flayer = fourier.FourierCube(coords=coords)
    # convert img_packed to pytorch tensor
    img_packed_tensor = torch.from_numpy(img_packed[np.newaxis, :, :])
    fourier_packed_num = np.squeeze(flayer(img_packed_tensor).numpy())

    # calculate the analytical FFT
    fourier_packed_an = utils.fourier_gaussian_lambda_arcsec(
        coords.packed_u_centers_2D, coords.packed_v_centers_2D, **gauss_kw
    )

    ikw = {"origin": "lower"}

    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(6, 8))
    im = ax[0, 0].imshow(fourier_packed_an.real, **ikw)
    plt.colorbar(im, ax=ax[0, 0])
    ax[0, 0].set_title("real")
    ax[0, 0].set_ylabel("analytical")
    im = ax[0, 1].imshow(fourier_packed_an.imag, **ikw)
    plt.colorbar(im, ax=ax[0, 1])
    ax[0, 1].set_title("imag")

    im = ax[1, 0].imshow(fourier_packed_num.real, **ikw)
    plt.colorbar(im, ax=ax[1, 0])
    ax[1, 0].set_ylabel("numerical")
    im = ax[1, 1].imshow(fourier_packed_num.imag, **ikw)
    plt.colorbar(im, ax=ax[1, 1])

    diff_real = fourier_packed_an.real - fourier_packed_num.real
    diff_imag = fourier_packed_an.imag - fourier_packed_num.imag

    im = ax[2, 0].imshow(diff_real, **ikw)
    ax[2, 0].set_ylabel("difference")
    plt.colorbar(im, ax=ax[2, 0])
    im = ax[2, 1].imshow(diff_imag, **ikw)
    plt.colorbar(im, ax=ax[2, 1])

    fig.savefig(tmp_path / "fourier_packed.png", dpi=300)

    assert np.all(np.abs(diff_real) < 1e-12)
    assert np.all(np.abs(diff_imag) < 1e-12)
    plt.close("all")


def test_fourier_cube_grad(coords):
    # Test that we can calculate a gradient on a loss function using the Fourier layer

    img_packed = utils.sky_gaussian_arcsec(
        coords.packed_x_centers_2D, coords.packed_y_centers_2D, **gauss_kw
    )

    # calculated the packed FFT using the FourierLayer
    flayer = fourier.FourierCube(coords=coords)
    # convert img_packed to pytorch tensor
    img_packed_tensor = torch.tensor(img_packed[np.newaxis, :, :], requires_grad=True)

    # calculated the packed FFT using the FourierLayer
    flayer = fourier.FourierCube(coords=coords)

    output = flayer(img_packed_tensor)
    loss = torch.sum(torch.abs(output))

    loss.backward()


def test_instantiate_nufft(coords):
    fourier.NuFFT(coords=coords, nchan=1)


def test_instantiate_nufft_cached_single_chan(coords, baselines_1D):
    # load some data
    uu, vv = baselines_1D

    # should assume everything is the same_uv
    layer = fourier.NuFFTCached(
        coords=coords, nchan=1, uu=uu, vv=vv, sparse_matrices=False
    )
    assert layer.same_uv

    # should assume everything is the same_uv. Uses sparse_matrices as default
    layer = fourier.NuFFTCached(coords=coords, nchan=1, uu=uu, vv=vv)
    assert layer.same_uv


def test_instantiate_nufft_cached_multi_chan(coords, baselines_1D):
    # load some data
    uu, vv = baselines_1D

    # should still assume that the uv is the same, since uu and vv are single-channel
    layer = fourier.NuFFTCached(
        coords=coords, nchan=10, uu=uu, vv=vv, sparse_matrices=False
    )
    assert layer.same_uv

    # should still assume that the uv is the same, since uu and vv are single-channel
    # should use sparse_matrices as default
    layer = fourier.NuFFTCached(coords=coords, nchan=10, uu=uu, vv=vv)
    assert layer.same_uv


def test_predict_vis_nufft(coords, baselines_1D):
    # just see that we can load the layer and get something through without error
    # for a very simple blank function

    # load some data
    uu, vv = baselines_1D

    nchan = 10

    # we have a multi-channel cube, but only sent single-channel uu and vv
    # coordinates. The expectation is that TorchKbNufft will parallelize these

    layer = fourier.NuFFT(coords=coords, nchan=nchan)

    # predict the values of the cube at the u,v locations
    blank_packed_img = torch.zeros((nchan, coords.npix, coords.npix))
    output = layer(blank_packed_img, uu, vv)

    # make sure we got back the number of visibilities we expected
    assert output.shape == (nchan, len(uu))

    # if the image cube was filled with zeros, then we should make sure this is true
    assert output.detach().numpy() == approx(
        np.zeros((nchan, len(uu)), dtype=np.complex128)
    )


def test_predict_vis_nufft_cached(coords, baselines_1D):
    # just see that we can load the layer and get something through without error
    # for a very simple blank function

    # load some data
    uu, vv = baselines_1D

    nchan = 10

    # we have a multi-channel cube, but sent only single-channel uu and vv
    # coordinates. The expectation is that TorchKbNufft will parallelize these
    layer = fourier.NuFFTCached(coords=coords, nchan=nchan, uu=uu, vv=vv)

    # predict the values of the cube at the u,v locations
    blank_packed_img = torch.zeros((nchan, coords.npix, coords.npix), dtype=torch.double)
    output = layer(blank_packed_img)

    # make sure we got back the number of visibilities we expected
    assert output.shape == (nchan, len(uu))

    # if the image cube was filled with zeros, then we should make sure this is true
    assert output.detach().numpy() == approx(
        np.zeros((nchan, len(uu)))
    )


def test_nufft_cached_predict_GPU(coords, baselines_1D):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        return

    # just see that we can load the layer and get something through without error
    # for a very simple blank function

    # load some data
    uu, vv = baselines_1D

    nchan = 10

    # we have a multi-channel cube, but only sent single-channel uu and vv
    # coordinates. The expectation is that TorchKbNufft will parallelize these

    layer = fourier.NuFFTCached(coords=coords, nchan=nchan, uu=uu, vv=vv).to(
        device=device
    )

    # predict the values of the cube at the u,v locations
    blank_packed_img = torch.zeros((nchan, coords.npix, coords.npix)).to(device=device)
    output = layer(blank_packed_img)

    # make sure we got back the number of visibilities we expected
    assert output.shape == (nchan, len(uu))

    # if the image cube was filled with zeros, then we should make sure this is true
    assert output.cpu().detach().numpy() == approx(
        np.zeros((nchan, len(uu)), dtype=np.complex128)
    )

def plot_nufft_comparison(uu, vv, an_output, num_output, path):
    """Plot and save a figure comparing the analytic and numerical FT points.
    """

    qq = utils.torch2npy(torch.hypot(uu, vv)) * 1e-6

    diff = num_output - an_output
    
    fig, ax = plt.subplots(nrows=4, sharex=True, figsize=(7,5))
    ax[0].scatter(qq, an_output.real, s=3, label="analytic")
    ax[0].scatter(qq, num_output.real, s=1, label="NuFFT")
    ax[0].set_ylabel("Real")
    ax[0].legend()

    ax[1].scatter(qq, diff.real, s=1, c="k")
    ax[1].set_ylabel("diff Real")

    ax[2].scatter(qq, an_output.imag, s=3)
    ax[2].scatter(qq, num_output.imag, s=1)
    ax[2].set_ylabel("Imag")

    ax[3].scatter(qq, diff.imag, s=1, c="k")
    ax[3].set_ylabel("diff Imag")
    ax[3].set_xlabel(r"$q$ [M$\lambda$]")

    fig.subplots_adjust(hspace=0.2, left=0.15, right=0.85, top=0.92)
    fig.suptitle("NuFFT Accuracy compared to analytic")
    fig.savefig(path, dpi=300)



def test_nufft_accuracy_single_chan(coords, baselines_1D, tmp_path):
    """Create a single-channel ImageCube using an analytic function for which we know 
    the true FT. 
    Then use the NuFFT to FT and sample that image.
    Plot both and their difference.
    Assert that the NuFFT samples and the analytic FT samples are close.
    """

    uu, vv = baselines_1D

    # NuFFT layer to perform interpolations to these points
    layer = fourier.NuFFT(coords=coords, nchan=1)

    img_packed = utils.sky_gaussian_arcsec(
        coords.packed_x_centers_2D, coords.packed_y_centers_2D, **gauss_kw
    )
    img_packed_tensor = torch.tensor(img_packed[np.newaxis, :, :], requires_grad=True)

    # use the NuFFT to predict the values of the cube at the u,v locations
    num_output = layer(img_packed_tensor, uu, vv)[0]  # take the channel dim out
    num_output = utils.torch2npy(num_output)

    # calculate the values analytically
    an_output = utils.fourier_gaussian_lambda_arcsec(uu, vv, **gauss_kw)
    an_output = utils.torch2npy(an_output)

    plot_nufft_comparison(uu, vv, an_output, num_output, tmp_path / "nufft_comparison.png")
    
    # threshold based on visual inspection of plot
    assert num_output == approx(an_output, abs=2.5e-6)


def test_nufft_cached_accuracy_single_chan(coords, baselines_1D, tmp_path):
    # create a single-channel ImageCube using a function we know the true FT analytically
    # use NuFFT to FT and sample that image
    # assert that the NuFFT samples and the analytic FT samples are close

    # load some data
    uu, vv = baselines_1D

    # create a NuFFT layer to perform interpolations to these points
    layer = fourier.NuFFTCached(coords=coords, nchan=1, uu=uu, vv=vv)

    img_packed = utils.sky_gaussian_arcsec(
        coords.packed_x_centers_2D, coords.packed_y_centers_2D, **gauss_kw
    )
    img_packed_tensor = torch.tensor(img_packed[np.newaxis, :, :], requires_grad=True, dtype=torch.double)

    # use the NuFFT to predict the values of the cube at the u,v locations
    num_output = layer(img_packed_tensor)[0]  # take the channel dim out
    num_output = utils.torch2npy(num_output)

    # calculate the values analytically
    an_output = utils.fourier_gaussian_lambda_arcsec(uu, vv, **gauss_kw)
    an_output = utils.torch2npy(an_output)

    plot_nufft_comparison(uu, vv, an_output, num_output, tmp_path / "nufft_cached_comparison.png")

    # threshold based on visual inspection of plot
    assert num_output == approx(an_output, abs=2e-8)


def test_nufft_cached_accuracy_coil_broadcast(coords, baselines_1D, tmp_path):
    # create a multi-channel ImageCube using a function we know the true FT analytically
    # use NuFFT to FT and sample that image
    # assert that the NuFFT samples and the analytic FT samples are close

    # load some data
    uu, vv = baselines_1D
    nchan = 5

    # create a NuFFT layer to perform interpolations to these points
    # since image is multi-channel but uu and vv are single-channel visibilities,
    # this should use the coil dimension of NuFFT to do the broadcasting
    layer = fourier.NuFFTCached(coords=coords, nchan=nchan, uu=uu, vv=vv)

    img_packed = utils.sky_gaussian_arcsec(
        coords.packed_x_centers_2D, coords.packed_y_centers_2D, **gauss_kw
    )

    # broadcast to 5 channels -- the image will be the same for each
    img_packed_tensor = torch.tensor(
        img_packed[np.newaxis, :, :] * np.ones((nchan, coords.npix, coords.npix)),
        requires_grad=True, dtype=torch.double
    )

    # use the NuFFT to predict the values of the cube at the u,v locations
    num_output = layer(img_packed_tensor)
    num_output = utils.torch2npy(num_output)
    
    # plot a single channel, to check
    ichan = 1

    # calculate the values analytically, for a single channel
    an_output = utils.fourier_gaussian_lambda_arcsec(uu, vv, **gauss_kw)
    an_output = utils.torch2npy(an_output)

    plot_nufft_comparison(
        uu, vv, an_output, num_output[ichan], tmp_path / "nufft_cached_comparison.png"
    )

    # loop through each channel and assert that things are the same
    for i in range(nchan):
        # threshold based on visual inspection of plot for single channel
        assert num_output[i] == approx(an_output, abs=2e-8)


def test_nufft_cached_accuracy_batch_broadcast(coords, baselines_2D_t, tmp_path):
    # create a single-channel ImageCube using a function we know the true FT analytically
    # use NuFFT to FT and sample that image
    # assert that the NuFFT samples and the analytic FT samples are close

    # load some multi-channel data
    uu, vv = baselines_2D_t
    nchan = uu.shape[0]

    # create a NuFFT layer to perform interpolations to these points
    # uu and vv are multidimensional, so we should set `sparse_matrices=False`
    # to avoid triggering a warning
    layer = fourier.NuFFTCached(
        coords=coords, nchan=nchan, uu=uu, vv=vv, sparse_matrices=False
    )

    img_packed = utils.sky_gaussian_arcsec(
        coords.packed_x_centers_2D, coords.packed_y_centers_2D, **gauss_kw
    )

    # broadcast to all channels -- the image will be the same for each
    img_packed_tensor = torch.tensor(
        img_packed[np.newaxis, :, :] * np.ones((nchan, coords.npix, coords.npix)),
        requires_grad=True,
    )

    # use the NuFFT to predict the values of the cube at the u,v locations
    num_output = layer(img_packed_tensor)
    num_output = utils.torch2npy(num_output)

    # plot a single channel, to check
    ichan = 1

    an_output = utils.fourier_gaussian_lambda_arcsec(uu[ichan], vv[ichan], **gauss_kw)
    an_output = utils.torch2npy(an_output)

    plot_nufft_comparison(
        uu[ichan], vv[ichan], an_output, num_output[ichan], tmp_path / "nufft_cached_comparison.png"
    )

    # loop through each channel and assert that things are the same
    for i in range(nchan):
        # calculate the values analytically for this channel
        an_output = utils.fourier_gaussian_lambda_arcsec(uu[i], vv[i], **gauss_kw)

        # using table-based interpolation, so the accuracy bar is lower
        # threshold based on visual inspection of plot of single channel
        assert num_output[i] == approx(an_output, abs=3e-6)

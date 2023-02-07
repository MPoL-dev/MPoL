from mpol.crossval import CrossValidate, DartboardSplitGridded, RandomCellSplitGridded
from mpol.datasets import Dartboard


def test_crossvalclass_split_dartboard(coords, gridder, dataset, generic_parameters):
    # using the CrossValidate class, split a dataset into train/test subsets 
    # using 'dartboard' splitter

    crossval_pars = generic_parameters["crossval_pars"]
    crossval_pars["split_method"] = "dartboard"

    cross_validator = CrossValidate(coords, gridder, **crossval_pars)
    cross_validator.split_dataset(dataset)


def test_crossvalclass_split_randomcell(coords, gridder, dataset, generic_parameters):
    # using the CrossValidate class, split a dataset into train/test subsets 
    # using 'random_cell' splitter

    crossval_pars = generic_parameters["crossval_pars"]
    cross_validator = CrossValidate(coords, gridder, **crossval_pars)
    cross_validator.split_dataset(dataset)


def test_crossvalclass_kfold(coords, gridder, dataset, generic_parameters):
    # using the CrossValidate class, perform k-fold cross-validation
 
    crossval_pars = generic_parameters["crossval_pars"]
    # reset some keys to bypass functionality tested elsewhere and speed up test
    crossval_pars["lambda_guess"] = None
    crossval_pars["epochs"] = 11

    cross_validator = CrossValidate(coords, gridder, **crossval_pars)
    cv_score, all_scores, loss_histories = cross_validator.run_crossval(dataset)


def test_dartboardsplit_init(crossvalidation_products):
    coords, dataset = crossvalidation_products

    dartboard = Dartboard(coords=coords)

    # create cross validator through passing dartboard
    DartboardSplitGridded(dataset, 5, dartboard=dartboard)

    # create cross validator through implicit creation of dartboard
    DartboardSplitGridded(dataset, 5)


def test_hermitian_mask_k(crossvalidation_products, tmp_path):
    coords, dataset = crossvalidation_products

    dartboard = Dartboard(coords=coords)
    chan = 4

    # split these into k samples
    k = 5
    cv = DartboardSplitGridded(dataset, k, dartboard=dartboard)

    # get the split list indices
    indices_l0 = cv.k_split_cell_list[0]

    # create a new mask from this
    dartboard_mask = dartboard.build_grid_mask_from_cells(indices_l0)

    # use this mask to index the dataset
    masked_dataset = copy.deepcopy(dataset)
    masked_dataset.add_mask(dartboard_mask)

    # get updated q and phi values
    qs = masked_dataset.coords.packed_q_centers_2D[masked_dataset.mask[chan]]
    phis = masked_dataset.coords.packed_phi_centers_2D[masked_dataset.mask[chan]]

    ind = phis <= np.pi

    fig, ax = plt.subplots(nrows=1)

    ax.plot(qs[ind], phis[ind], "o", ms=3)
    ax.plot(qs[~ind], phis[~ind] - np.pi, "o", ms=1)
    fig.savefig(tmp_path / "hermitian.png", dpi=300)


def test_dartboardsplit_iterate_masks(crossvalidation_products, tmp_path):
    coords, dataset = crossvalidation_products

    dartboard = Dartboard(coords=coords)

    # create cross validator through passing dartboard
    k = 5
    chan = 4
    cv = DartboardSplitGridded(dataset, k, dartboard=dartboard)

    fig, ax = plt.subplots(nrows=k, ncols=2, figsize=(6, 12))

    for k, (train, test) in enumerate(cv):

        ax[k, 0].imshow(
            np.fft.fftshift(train.mask[chan].detach().numpy()),
            interpolation="none",
        )
        ax[k, 1].imshow(
            np.fft.fftshift(test.mask[chan].detach().numpy()),
            interpolation="none",
        )

    ax[0, 0].set_title("train")
    ax[0, 1].set_title("test")
    fig.savefig(tmp_path / "masks", dpi=300)


def test_dartboardsplit_iterate_images(crossvalidation_products, tmp_path):
    coords, dataset = crossvalidation_products

    dartboard = Dartboard(coords=coords)

    # create cross validator through passing dartboard
    k = 5
    chan = 4
    cv = DartboardSplitGridded(dataset, k, dartboard=dartboard)

    # visualize dirty images
    # create mock fourier layer
    flayer = fourier.FourierCube(coords=coords)
    flayer.forward(torch.zeros(dataset.nchan, coords.npix, coords.npix))

    fig, ax = plt.subplots(nrows=k, ncols=4, figsize=(12, 12))

    for k, (train, test) in enumerate(cv):

        rtrain = connectors.GriddedResidualConnector(flayer, train)
        rtest = connectors.GriddedResidualConnector(flayer, test)

        train_chan = utils.packed_cube_to_sky_cube(rtrain.forward())[chan]
        test_chan = utils.packed_cube_to_sky_cube(rtest.forward())[chan]

        im = ax[k, 0].imshow(
            train_chan.real.detach().numpy(), interpolation="none", origin="lower"
        )
        plt.colorbar(im, ax=ax[k, 0])

        im = ax[k, 1].imshow(
            train_chan.imag.detach().numpy(), interpolation="none", origin="lower"
        )
        plt.colorbar(im, ax=ax[k, 1])

        im = ax[k, 2].imshow(
            test_chan.real.detach().numpy(), interpolation="none", origin="lower"
        )
        plt.colorbar(im, ax=ax[k, 2])

        im = ax[k, 3].imshow(
            test_chan.imag.detach().numpy(), interpolation="none", origin="lower"
        )
        plt.colorbar(im, ax=ax[k, 3])

    ax[0, 0].set_title("train")
    ax[0, 2].set_title("test")
    fig.savefig(tmp_path / "images", dpi=300)


# TODO: add gpu test
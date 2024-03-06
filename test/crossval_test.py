import copy

import matplotlib.pyplot as plt
import numpy as np

# from mpol.crossval import CrossValidate, DartboardSplitGridded, RandomCellSplitGridded
from mpol.crossval import DartboardSplitGridded, RandomCellSplitGridded
from mpol.datasets import Dartboard

# def test_crossvalclass_split_dartboard(coords, imager, dataset, generic_parameters):
#     # using the CrossValidate class, split a dataset into train/test subsets
#     # using 'dartboard' splitter

#     crossval_pars = generic_parameters["crossval_pars"]
#     crossval_pars["split_method"] = "dartboard"

#     cross_validator = CrossValidate(coords, imager, **crossval_pars)
#     cross_validator.split_dataset(dataset)


# def test_crossvalclass_split_dartboard_1kfold(
#     coords, imager, dataset, generic_parameters
# ):
#     # using the CrossValidate class, split a dataset into train/test subsets
#     # using 'dartboard' splitter with only 1 k-fold; check that the train set
#     # has ~80% of the model visibilities

#     crossval_pars = generic_parameters["crossval_pars"]
#     crossval_pars["split_method"] = "dartboard"
#     crossval_pars["kfolds"] = 1

#     cross_validator = CrossValidate(coords, imager, **crossval_pars)
#     split_iterator = cross_validator.split_dataset(dataset)

#     for train_set, test_set in split_iterator:
#         ntrain = len(train_set.vis_indexed)
#         ntest = len(test_set.vis_indexed)

#     ratio = ntrain / (ntrain + ntest)

#     np.testing.assert_allclose(ratio, 0.8, atol=0.05)


# def test_crossvalclass_split_randomcell(coords, imager, dataset, generic_parameters):
#     # using the CrossValidate class, split a dataset into train/test subsets
#     # using 'random_cell' splitter

#     crossval_pars = generic_parameters["crossval_pars"]
#     cross_validator = CrossValidate(coords, imager, **crossval_pars)
#     cross_validator.split_dataset(dataset)


# def test_crossvalclass_split_diagnostics_fig(
#     coords, imager, dataset, generic_parameters, tmp_path
# ):
#     # using the CrossValidate class, split a dataset into train/test subsets
#     # using 'random_cell' splitter, then generate the split diagnostic figure

#     crossval_pars = generic_parameters["crossval_pars"]
#     cross_validator = CrossValidate(coords, imager, **crossval_pars)
#     split_iterator = cross_validator.split_dataset(dataset)
#     split_fig, split_axes = split_diagnostics_fig(split_iterator)
#     split_fig.savefig(tmp_path / "split_diagnostics_fig.png", dpi=300)
#     plt.close("all")


# def test_crossvalclass_kfold(coords, imager, dataset, generic_parameters):
#     # using the CrossValidate class, perform k-fold cross-validation

#     crossval_pars = generic_parameters["crossval_pars"]
#     # reset some keys to bypass functionality tested elsewhere and speed up test
#     crossval_pars["regularizers"] = {}
#     crossval_pars["epochs"] = 11

#     cross_validator = CrossValidate(coords, imager, **crossval_pars)
#     cross_validator.run_crossval(dataset)


def test_randomcellsplit(dataset, generic_parameters):
    pars = generic_parameters["crossval_pars"]
    RandomCellSplitGridded(dataset, pars["kfolds"], pars["seed"])


def test_dartboardsplit_init(coords, dataset):
    dartboard = Dartboard(coords=coords)

    # create cross validator through passing dartboard
    DartboardSplitGridded(dataset, 5, dartboard=dartboard)

    # create cross validator through implicit creation of dartboard
    DartboardSplitGridded(dataset, 5)


def test_hermitian_mask_k(coords, dataset, tmp_path):
    dartboard = Dartboard(coords=coords)
    chan = 1

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


def test_dartboardsplit_iterate_masks(coords, dataset, tmp_path):
    dartboard = Dartboard(coords=coords)

    # create cross validator through passing dartboard
    k = 5
    chan = 1
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

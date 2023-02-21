import numpy as np
import torch
from mpol import (
    losses,
    coordinates,
    images,
    precomposed,
    gridding,
    datasets,
    connectors,
    utils,
)
from astropy.utils.data import download_file
from ray import tune
import matplotlib.pyplot as plt

# We want to split these
# because otherwise the data loading routines will be rehashed several times.


def train(
    model, dataset, optimizer, config, device, writer=None, report=False, logevery=50
):
    """
    Args:
        model: neural net model
        dataset: to use to train against
        optimizer: tied to model parameters and used to take a step
        config: dictionary including epochs and hyperparameters.
        device: "cpu" or "cuda"
        writer: tensorboard writer object
    """
    model = model.to(device)
    model.train()
    dataset = dataset.to(device)
    residuals = connectors.GriddedResidualConnector(model.fcube, dataset)
    residuals.to(device)

    for iteration in range(config["epochs"]):

        optimizer.zero_grad()
        vis = model()
        sky_cube = model.icube.sky_cube

        loss = (
            losses.nll_gridded(vis, dataset)
            + config["lambda_sparsity"] * losses.sparsity(sky_cube)
            + config["lambda_TV"] * losses.TV_image(sky_cube)
            + config["entropy"] * losses.entropy(sky_cube, config["prior_intensity"])
        )

        if (iteration % logevery == 0) and writer is not None:
            writer.add_scalar("loss", loss.item(), iteration)
            writer.add_figure("image", log_figure(model, residuals), iteration)

        loss.backward()
        optimizer.step()

    if report:
        tune.report(loss=loss.item())

    return loss.item()


def test(model, dataset, device):
    model = model.to(device)
    model.eval()
    dataset = dataset.to(device)
    vis = model()
    loss = losses.nll_gridded(vis, dataset)
    return loss.item()


def cross_validate(model, config, device, k_fold_datasets, MODEL_PATH, writer=None):

    test_scores = []

    for k_fold, (train_dset, test_dset) in enumerate(k_fold_datasets):

        # reset model
        model.load_state_dict(torch.load(MODEL_PATH))

        # create a new optimizer for this k_fold
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

        # train for a while
        train(model, train_dset, optimizer, config, device, writer=writer)
        # evaluate the test metric
        test_scores.append(test(model, test_dset, device))

    # aggregate all test scores and sum to evaluate cross val metric
    test_score = np.sum(np.array(test_scores))

    # log to ray tune
    tune.report(cv_score=test_score)

    return test_score


def log_figure(model, residuals):
    """
    Create a matplotlib figure showing the current image state.

    Args:
        model: neural net model
    """

    # populate residual connector
    residuals()

    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))
    im = ax[0, 0].imshow(
        np.squeeze(model.icube.sky_cube.detach().cpu().numpy()),
        origin="lower",
        interpolation="none",
        extent=model.icube.coords.img_ext,
    )
    plt.colorbar(im, ax=ax[0, 0])

    im = ax[0, 1].imshow(
        np.squeeze(residuals.sky_cube.detach().cpu().numpy()),
        origin="lower",
        interpolation="none",
        extent=residuals.coords.img_ext,
    )
    plt.colorbar(im, ax=ax[0, 1])

    im = ax[1, 0].imshow(
        np.squeeze(torch.log(model.fcube.ground_amp.detach()).cpu().numpy()),
        origin="lower",
        interpolation="none",
        extent=residuals.coords.vis_ext,
    )
    plt.colorbar(im, ax=ax[1, 0])

    im = ax[1, 1].imshow(
        np.squeeze(torch.log(residuals.ground_amp.detach()).cpu().numpy()),
        origin="lower",
        interpolation="none",
        extent=residuals.coords.vis_ext,
    )
    plt.colorbar(im, ax=ax[1, 1])

    return fig

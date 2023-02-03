import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim
from torch.utils.tensorboard import SummaryWriter

from mpol import losses, precomposed
from mpol.constants import *


def test_standalone_init_train(coords, dataset):
    # not using TrainTest class, 
    # configure a class to train with and test that it initializes

    nchan = dataset.nchan
    rml = precomposed.SimpleNet(coords=coords, nchan=nchan)

    vis = rml.forward()

    rml.zero_grad()

    # calculate a loss
    loss = losses.nll_gridded(vis, dataset)

    # calculate gradients of parameters
    loss.backward()

    print(rml.bcube.base_cube.grad)


def test_standalone_train_loop(coords, dataset_cont, tmp_path):
    # not using TrainTest class, 
    # set everything up to run on a single channel
    # and run a few iterations

    nchan = 1
    rml = precomposed.SimpleNet(coords=coords, nchan=nchan)

    optimizer = torch.optim.SGD(rml.parameters(), lr=0.001)

    for i in range(50):
        rml.zero_grad()

        # get the predicted model
        vis = rml.forward()

        # calculate a loss
        loss = losses.nll_gridded(vis, dataset_cont)

        # calculate gradients of parameters
        loss.backward()

        # update the model parameters
        optimizer.step()

    # let's see what one channel of the image looks like
    fig, ax = plt.subplots(nrows=1)
    ax.imshow(
        np.squeeze(rml.icube.cube.detach().numpy()),
        origin="lower",
        interpolation="none",
        extent=rml.icube.coords.img_ext,
    )
    fig.savefig(tmp_path / "trained.png", dpi=300)
    plt.close("all")


def test_tensorboard(coords, dataset_cont, tmp_path):
    # not using TrainTest class, 
    # set everything up to run on a single channel and then
    # test the writer function

    nchan = 1
    rml = precomposed.SimpleNet(coords=coords, nchan=nchan)

    optimizer = torch.optim.SGD(rml.parameters(), lr=0.001)

    writer = SummaryWriter()

    for i in range(50):
        rml.zero_grad()

        # get the predicted model
        vis = rml.forward()

        # calculate a loss
        loss = losses.nll_gridded(vis, dataset_cont)

        writer.add_scalar("loss", loss.item(), i)

        # calculate gradients of parameters
        loss.backward()

        # update the model parameters
        optimizer.step()

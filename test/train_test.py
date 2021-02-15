import pytest
import numpy as np
import torch
import torch.optim
import matplotlib.pyplot as plt
from mpol import connectors, images, losses
from mpol.constants import *


# configure a class to train with

# https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
class RML(torch.nn.Module):
    def __init__(self, coords, nchan, base_cube, dataset):
        super().__init__()
        self.bcube = images.BaseCube(coords=coords, nchan=nchan, base_cube=base_cube)
        self.icube = images.ImageCube(coords=coords, nchan=nchan, passthrough=True)
        self.flayer = images.FourierCube(coords=coords)
        self.dcon = connectors.DatasetConnector(self.flayer, dataset)

    def forward(self):
        # produce model visibilities
        x = self.bcube.forward()
        x = self.icube.forward(x)
        vis = self.flayer.forward(x)
        model_samples = self.dcon.forward(vis)
        return model_samples


# currently segfaults on 3.9
# https://github.com/pytorch/pytorch/issues/50014
def test_init_train_class(coords, dataset):

    nchan = dataset.nchan
    rml = RML(coords, nchan, None, dataset)

    model_visibilities = rml.forward()

    rml.zero_grad()

    # calculate a loss
    loss = losses.nll(model_visibilities, dataset.vis_indexed, dataset.weight_indexed)

    # calculate gradients of parameters
    loss.backward()

    print(rml.bcube.base_cube.grad)


def test_train_loop(coords, dataset_cont, tmp_path):

    # set everything up to run on a single channel

    nchan = 1
    rml = RML(coords, nchan, None, dataset_cont)

    optimizer = torch.optim.SGD(rml.parameters(), lr=0.001)

    for i in range(300):
        rml.zero_grad()

        # get the predicted model
        model_visibilities = rml.forward()

        # calculate a loss
        loss = losses.nll(
            model_visibilities, dataset_cont.vis_indexed, dataset_cont.weight_indexed
        )

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

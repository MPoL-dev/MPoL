import pytest
import numpy as np
import torch
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


def test_init_train_class(coords, dataset):

    nchan = dataset.nchan
    rml = RML(coords, nchan, None, dataset)

    model_visibilities = rml.forward()

    rml.zero_grad()

    # calculate a loss
    loss = losses.nll(model_visibilities, dataset.vis_indexed, dataset.weight_indexed)

    # calculate gradients of parameters
    # loss.backward()

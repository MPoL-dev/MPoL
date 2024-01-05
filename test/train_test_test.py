import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim
from torch.utils.tensorboard import SummaryWriter

from mpol import losses, precomposed
# from mpol.plot import train_diagnostics_fig
# from mpol.training import TrainTest, train_to_dirty_image
from mpol.utils import torch2npy


# def test_traintestclass_training(coords, imager, dataset, generic_parameters):
#     # using the TrainTest class, run a training loop without regularizers
#     nchan = dataset.nchan
#     model = precomposed.GriddedNet(coords=coords, nchan=nchan)

#     train_pars = generic_parameters["train_pars"]
    
#     # no regularizers
#     train_pars["regularizers"] = {}

#     learn_rate = generic_parameters["crossval_pars"]["learn_rate"]

#     optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

#     trainer = TrainTest(imager=imager, optimizer=optimizer, **train_pars)
#     loss, loss_history = trainer.train(model, dataset)


# def test_traintestclass_training_scheduler(coords, imager, dataset, generic_parameters):
#     # using the TrainTest class, run a training loop with regularizers, 
#     # using the learning rate scheduler
#     nchan = dataset.nchan
#     model = precomposed.GriddedNet(coords=coords, nchan=nchan)

#     train_pars = generic_parameters["train_pars"]

#     learn_rate = generic_parameters["crossval_pars"]["learn_rate"]

#     optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

#     # use a scheduler
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.995)
#     train_pars["scheduler"] = scheduler

#     trainer = TrainTest(imager=imager, optimizer=optimizer, **train_pars)
#     loss, loss_history = trainer.train(model, dataset)


# def test_traintestclass_training_guess(coords, imager, dataset, generic_parameters):
#     # using the TrainTest class, run a training loop with regularizers,
#     # with a call to the regularizer strength guesser
#     nchan = dataset.nchan
#     model = precomposed.GriddedNet(coords=coords, nchan=nchan)

#     train_pars = generic_parameters["train_pars"] 

#     learn_rate = generic_parameters["crossval_pars"]["learn_rate"]

#     train_pars['regularizers']['entropy']['guess'] = True 

#     optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

#     trainer = TrainTest(imager=imager, optimizer=optimizer, **train_pars)
#     loss, loss_history = trainer.train(model, dataset)


# def test_traintestclass_train_diagnostics_fig(coords, imager, dataset, generic_parameters, tmp_path):
#     # using the TrainTest class, run a training loop, 
#     # and generate the train diagnostics figure 
#     nchan = dataset.nchan
#     model = precomposed.GriddedNet(coords=coords, nchan=nchan)

#     train_pars = generic_parameters["train_pars"]
#     # bypass TrainTest.loss_lambda_guess
#     train_pars["regularizers"] = {}

#     learn_rate = generic_parameters["crossval_pars"]["learn_rate"]

#     optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

#     trainer = TrainTest(imager=imager, optimizer=optimizer, **train_pars)
#     loss, loss_history = trainer.train(model, dataset)

#     learn_rates = np.repeat(learn_rate, len(loss_history))

#     old_mod_im = torch2npy(model.icube.sky_cube[0])

#     train_fig, train_axes = train_diagnostics_fig(model, 
#                                                   losses=loss_history, 
#                                                   learn_rates=learn_rates,
#                                                   fluxes=np.zeros(len(loss_history)),
#                                                   old_model_image=old_mod_im
#                                                   )
#     train_fig.savefig(tmp_path / "train_diagnostics_fig.png", dpi=300)
#     plt.close("all")


# def test_traintestclass_testing(coords, imager, dataset, generic_parameters):
#     # using the TrainTest class, perform a call to test
#     nchan = dataset.nchan
#     model = precomposed.GriddedNet(coords=coords, nchan=nchan)

#     learn_rate = generic_parameters["crossval_pars"]["learn_rate"]

#     optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

#     trainer = TrainTest(imager=imager, optimizer=optimizer)
#     trainer.test(model, dataset)


def test_standalone_init_train(coords, dataset):
    # not using TrainTest class, 
    # configure a class to train with and test that it initializes

    nchan = dataset.nchan
    rml = precomposed.GriddedNet(coords=coords, nchan=nchan)

    vis = rml()

    rml.zero_grad()

    # calculate a loss
    loss = losses.r_chi_squared_gridded(vis, dataset)

    # calculate gradients of parameters
    loss.backward()

    print(rml.bcube.base_cube.grad)


def test_standalone_train_loop(coords, dataset_cont, tmp_path):
    # not using TrainTest class, 
    # set everything up to run on a single channel
    # and run a few iterations

    nchan = 1
    rml = precomposed.GriddedNet(coords=coords, nchan=nchan)

    optimizer = torch.optim.SGD(rml.parameters(), lr=0.001)

    for i in range(50):
        rml.zero_grad()

        # get the predicted model
        vis = rml()

        # calculate a loss
        loss = losses.r_chi_squared_gridded(vis, dataset_cont)

        # calculate gradients of parameters
        loss.backward()

        # update the model parameters
        optimizer.step()

    # let's see what one channel of the image looks like
    fig, ax = plt.subplots(nrows=1)
    ax.imshow(
        np.squeeze(torch2npy(rml.icube.packed_cube)),
        origin="lower",
        interpolation="none",
        extent=rml.icube.coords.img_ext,
    )
    fig.savefig(tmp_path / "trained.png", dpi=300)
    plt.close("all")


# def test_train_to_dirty_image(coords, dataset, imager):
#     # run a training loop against a dirty image
#     nchan = dataset.nchan
#     model = precomposed.GriddedNet(coords=coords, nchan=nchan)

#     train_to_dirty_image(model, imager, niter=10)


def test_tensorboard(coords, dataset_cont):
    # not using TrainTest class, 
    # set everything up to run on a single channel and then
    # test the writer function

    nchan = 1
    rml = precomposed.GriddedNet(coords=coords, nchan=nchan)

    optimizer = torch.optim.SGD(rml.parameters(), lr=0.001)

    writer = SummaryWriter()

    for i in range(50):
        rml.zero_grad()

        # get the predicted model
        vis = rml()

        # calculate a loss
        loss = losses.r_chi_squared_gridded(vis, dataset_cont)

        writer.add_scalar("loss", loss.item(), i)

        # calculate gradients of parameters
        loss.backward()

        # update the model parameters
        optimizer.step()

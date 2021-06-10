import torch
import matplotlib.pyplot as plt
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from common_data import model, dataset, residuals
from common_functions import train, log_figure


config = {'lr': 0.3, 'lambda_sparsity': 7.076022085822013e-05, 'lambda_TV': 0.00, 'entropy': 1e-03, 'prior_intensity': 1.597766235483388e-07, 'epochs': 1000}


# query to see if we have a GPU
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

tic = time.perf_counter()
# initialize model to trained dirty image
model.load_state_dict(torch.load("model.pt"))

# create an optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

# writer = SummaryWriter()
writer = None

# run the training loop
train(model, dataset, optimizer, config, device=device, writer=writer)
toc = time.perf_counter()

print("Elapsed time {:} s".format(toc - tic))

# save the visibility
fig = log_figure(model, residuals)
fig.savefig("residuals.png", dpi=300)

# save the image
fig, ax = plt.subplots(nrows=1, figsize=(8, 8))
im = ax.imshow(
    np.squeeze(model.icube.sky_cube.detach().cpu().numpy()),
    origin="lower",
    interpolation="none",
    extent=model.icube.coords.img_ext,
)
plt.colorbar(im)
fig.savefig("trained.png", dpi=300)


def scale(I):
    a = 0.02
    return np.arcsinh(I / a) / np.arcsinh(1 / a)


fig, ax = plt.subplots(nrows=1, figsize=(8, 8))
im = ax.imshow(
    scale(np.squeeze(model.icube.sky_cube.detach().cpu().numpy())),
    origin="lower",
    interpolation="none",
    extent=model.icube.coords.img_ext,
)
plt.colorbar(im)
fig.savefig("trained-arcsinh.png", dpi=300)

np.savez(
    "rml.npz",
    img=np.squeeze(model.icube.sky_cube.detach().cpu().numpy()),
    ext=model.icube.coords.img_ext,
)

model.icube.to_FITS(overwrite=True)

import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from common_data import gridder, model


# get the dirty image we want to optimize to
img, beam = gridder.get_dirty_image(weighting="briggs", robust=0.0, unit="Jy/arcsec^2")
# make img a tensor
dirty_image = torch.tensor(img.copy())

# predict the model that produces the dirty image
# create an optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.5)

# query to see if we have a GPU
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

model = model.to(device)
model.train()
dirty_image = dirty_image.to(device)
writer = SummaryWriter()

loss_fn = torch.nn.MSELoss()

for iteration in range(500):

    optimizer.zero_grad()

    model()
    sky_cube = model.icube.sky_cube

    loss = loss_fn(sky_cube, dirty_image)
    writer.add_scalar("loss", loss.item(), iteration)

    loss.backward()
    optimizer.step()

# save the model
torch.save(model.state_dict(), "model.pt")

fig, ax = plt.subplots(ncols=2, figsize=(7, 3.5))

im = ax[0].imshow(
    np.squeeze(dirty_image.detach().cpu().numpy()),
    origin="lower",
    interpolation="none",
    extent=model.icube.coords.img_ext,
)
plt.colorbar(im, ax=ax[0])

im = ax[1].imshow(
    np.squeeze(model.icube.sky_cube.detach().cpu().numpy()),
    origin="lower",
    interpolation="none",
    extent=model.icube.coords.img_ext,
)
plt.colorbar(im, ax=ax[1])

fig.savefig("optimized_dirty.png", dpi=300)

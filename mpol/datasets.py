import numpy as np
import torch
from torch.utils.data import Dataset

# custom dataset loader
class UVDataset(Dataset):
    def __init__(self, uu, vv, weights, data_real, data_imag, **kwargs):
        """
        PyTorch Dataset container for interferometric visibilities.

        Args:
            uu (2d array: (nchan, nvis)):  
            vv (2d array: (nchan, nvis)):

        Optional pre-gridding.
        """

        # assert that all vectors are 1D and the same length
        shape = uu.shape
        assert (
            len(shape) == 1
        ), "All dataset inputs must be 1D and have the same length."
        for a in [vv, weights, data_real, data_imag]:
            assert (
                a.shape == shape
            ), "All dataset inputs must be 1D and have the same length."

        self.uu = torch.tensor(uu, dtype=torch.double)  # klambda
        self.vv = torch.tensor(vv, dtype=torch.double)  # klambda
        self.weights = torch.tensor(weights, dtype=torch.double)  # 1/Jy^2
        self.re = torch.tensor(data_real, dtype=torch.double)  # Jy
        self.im = torch.tensor(data_imag, dtype=torch.double)  # Jy

        # TODO: store kwargs to do something for antenna self-cal

    def __getitem__(self, index):
        return (
            self.uu[index],
            self.vv[index],
            self.weights[index],
            self.re[index],
            self.im[index],
        )

    def __len__(self):
        return len(self.uu)

# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + nbsphinx="hidden"
# %matplotlib inline

# + nbsphinx="hidden"
# %run notebook_setup
# -

# # Tensors and Gradient Descent

# This tutorial covers how to initialize tensors, calculation of gradients, and the idea of a gradient descent.
#
# ## Introduction to Tensors
#
# Tensors are matrices, similar to numpy arrays, with the added benefit of calculations of gradients and optimization processes (more on that later). These are useful in RML imaging for the usual purposes of matrices (i.e. multiple variables pointing to one data point), for their automatic differentiation capabilities when cleaning up an image and their fast processing time.
#
# We'll start by importing the torch and numpy packages. Make sure you have [PyTorch installed](https://pytorch.org/get-started/locally/) on your device or virtual environment before proceeding.

import torch
import numpy as np
import matplotlib.pyplot as plt

# ### Initialization of Tensors
#
# There are several [ways to initialize a tensor](https://pytorch.org/docs/stable/tensors.html). A common method to create a tensor is from a numpy array:

# +
an_array = np.array([[1, 2], [3, 4]])
a_tensor = torch.tensor(an_array)  # creates tensor of same size as an_array

print(a_tensor)
# -

# Tensors have many parallels to numpy arrays. There are several [operations](https://pytorch.org/docs/stable/torch.html) that we can perform on tensors that we'd usually perform on numpy arrays. For example, we can multiply two numpy arrays and compare this with the multiplication of the corresponding PyTorch tensors:

# +
another_array = np.array([[5, 6, 7], [8, 9, 0]])  # create 2x3 array
another_tensor = torch.tensor(
    another_array
)  # create another tensor of same size as above array


# numpy array multiplication
prod_array = np.matmul(an_array, another_array)

# torch tensor multiplication
prod_tensor = torch.matmul(a_tensor, another_tensor)


print(f"Numpy array multiplication result: {prod_array}")
print(f"Torch tensor multiplication result: {prod_tensor}")


# -

# For the next section, we will create a tensor of a single value. Here we are setting <code> requires_grad = True </code>, we'll see why this is important in a moment.

x = torch.tensor(3.0, requires_grad=True)  # create a tensor with a single 3
x

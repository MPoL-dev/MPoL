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

# ## Calculation of Gradients

# ## Calculation of Gradients

# PyTorch also allows us to to calculate the gradients on tensors.
#
# Let's define some variable y in terms of x:

y = x ** 2

#  When we plug in <code> x </code> from the previous section we find that the output is what we would expect had we plugged <3.> into the function:

print(f"y: {y}")
print(f"x: {x}")

# Going back to our function, if we calculated the gradient by hand using calculus we would expect dy/dx = 2*x and the gradient evaluated for our original value <code> x </code> would be <6.>.
#
# We can see if PyTorch gets the same answer as us if we do:

y.backward()  # populates gradient (.grad) attributes of y with respect to all of its independent variables
x.grad  # returns the grad attribute (the gradient) of y with respect to x

# It works! But what does it mean? Instead of computing the derivative as we would by hand, the program is using a computational graph.

# First we create a tensor object x. This creates a [leaf](https://pytorch.org/docs/master/generated/torch.Tensor.is_leaf.html) on our computational graph. Right now, <code> x.grad = None </code> because we haven't inserted any code to tell the program to compute the gradient.

# ![Xleaf.png](attachment:Xleaf.png)

# Then we perform an operation on x to get y. We are traversing forward on the computational graph. Recall, when we first created the x tensor we set <code> requires_grad = True </code> This indicates that we want the computer to keep track of what operations we perform on x.

# ![Ynode.png](attachment:Ynode.png)

# When we want the gradient to be computed we use <code> y.backward() </code>. **Leaf variables** with <code> requires_grad = True </code> that are connected to y will have the <code> grad </code> property populated with the derivative of y with respect to that leaf variable. In this case, dy/dx. We are traversing backward through the computational graph.
#
# In the scope of this tutorial we are looking at the gradients of scalar tensors (only containing one element). In the case of non-tensor scalars the parameter [grad_tensors](https://pytorch.org/docs/stable/autograd.html) would need to be specified in the backward function in order to not receive an error.

# ![ybackward.png](attachment:ybackward.png)

# To see what x.grad is we must run <code> x.grad </code>

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

# ##### Multivariable functions
#
# This process can also be done on multivariable functions. We can start with three tensors:

a = torch.tensor(4.0, requires_grad=True)
b = torch.tensor(5.0, requires_grad=True)
c = torch.tensor(6.0, requires_grad=True)

# This gives us 3 leaves on our computational graph:

# ![MultivarLeaves.png](attachment:MultivarLeaves.png)

# We can create new variables by performing operations on a, b, and c:

d = a + b
e = 3 * c

#  Since we have <code> requires_grad = True </code>, these changes are tracked through the computational graph. This creates the two nodes on our computational graph:

# ![MultivarL1Nodes.png](attachment:MultivarL1Nodes.png)

# We can create one more variable by performing operations on d and e:

f = e - d
print(f"f: {f}")

# This creates what will be our final layer, with our last node, on our computational graph:

# ![MultivarL2Node.png](attachment:MultivarL2Node.png)

# We can calculate $\frac{\partial f}{\partial a}$, $\frac{\partial f}{\partial b}$, and $\frac{\partial f}{\partial c}$ by using f.backward(). Recall that <code>torch.autograd.backward()</code> will only compute the sum of gradients with respect to **leaf nodes**. This is accomplished using the chain rule.

# In the case of df/da, we see that this can be obtained by:
#
#
# $ \frac{\partial f}{\partial a} = \frac{\partial f}{\partial d} \frac{\partial d}{\partial a} $
#
#
# Where <code>torch.autograd.backward()</code>  obtains values for each of the partial derivatives along each of the edges in the computational graphs.

f.backward()

# ![MultivarGradPop.png](attachment:MultivarGradPop.png)

print(f"a.grad = {a.grad}")
print(f"b.grad = {b.grad}")
print(f"c.grad = {c.grad}")

# When we try <code> d.grad </code> we get a warning:

d.grad

# This is because <code> y.backward() </code> only populates grad attributes of **leaves** not nodes.
#
# Another thing to note, if there were multiple paths connecting a leaf to a tensor, the grad attribute would be calculated by (1) Multiplying the partial derivatives along each edge of a single path together (2) Adding up the products of each path together.  To see an example of this go [here](https://blog.paperspace.com/pytorch-101-understanding-graphs-and-automatic-differentiation/)

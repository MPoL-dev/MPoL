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

# # Introduction to PyTorch Tensors and Gradient Descent

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

# ## Calculating Gradients

# PyTorch provides a key functionality---the ability to calculate the gradients on tensors. Let's start by creating a tensor with a single value. Here we are setting <code> requires_grad = True </code>, we'll see why this is important in a moment.

x = torch.tensor(3.0, requires_grad=True)
x

# Let's define some variable y in terms of x:

y = x ** 2

#  We see that the value of ``y`` is as we expect---nothing too strange here.

print(f"y: {y}")
print(f"x: {x}")

# But what if we wanted to calculate the gradient of ``y`` with respect to ``x``? Using calculus, we find that the answer is dy/dx = 2*x. The derivative evaluated at ``x=3`` is ``6``.
#
# We can see if PyTorch gets the same answer as us if we do:

y.backward()  # populates gradient (.grad) attributes of y with respect to all of its independent variables
x.grad  # returns the grad attribute (the gradient) of y with respect to x

# PyTorch uses the concept of automatic differentiation to calculate the derivative. Instead of computing the derivative as we would by hand, the program is using a computational graph.

# ## Optimizing a Function with Gradient Descent
#
# If we were on the side of a hill in the dark and we wanted to get down to the bottom of a valley, how would we do it?
#
# We wouldn't be able to see all the way to the bottom of the valley, but we could feel which way is down based on where we are standing. We would take steps in the downward direction and we'd know when to stop when the ground felt flat.
#
# One other thing we'd have to consider is our step size. If we take very small steps in the direction of the descent, it will take us a longer time than if we take larger steps. However, if we take super long steps, we might completely miss the flat part of the valley, and start ascending the other side of the valley.

# We can look at the gradient descent from a more mathematical lense by looking at the graph z = t ** 2:

t = np.linspace(-5, 5, 100)
plt.plot(t, t ** 2)

# We start on the left hill at the point (-4, 16):

t = np.linspace(-5, 5, 100)
plt.plot(t, t ** 2)
plt.scatter(-4, 16)
plt.show()

# If we plot a tangent line at this point, we see which directions we can go:

# +
t = np.linspace(-5, 5, 100)
plt.plot(t, t ** 2)


plt.scatter(-4, 16)  # point
tp = -4
yp = 16

plt.plot(t, tp * 2 * (t - tp) + yp)  # tangent line

plt.xlim(xmin=-5, xmax=5)
plt.ylim(ymin=0, ymax=25)
plt.show()
# -

# We see we need to go down to go toward the minimum. We take a very small step in the direction of the steepest downward slope. When we take steps, we find the x coordinate of our new location by this equation:
#
# $x_{new} = x_{current} - \nabla y(x_{current}) * (step \: size)$
#
# where:
# - $x_{current}$ is our current x value
#
# - $\nabla y(x_{current})$ is the gradient at our current point
#
# - $(step \: size)$ is a value we choose that tells us scales our steps
#
# This makes sense because we start at some x value and we want to get to a new x value that is closer to the minimum. In the above plot, the gradient is negative. If we were to add the gradient to our current x value that would bring us further away from the minimum to a more negative x value. This is because the gradient points in the direction of the steepest ascent while we are looking to go in the direction of the steepest descent. This is why there is a negative in the equation.
#
#
# We will choose `step size = 0.1`:

# +
t = np.linspace(-5, 5, 100)
plt.plot(t, t ** 2)


plt.scatter(-4, 16)  # point

plt.scatter(-3.2, 10.24)  # second point
plt.arrow(-4, 16, 0.8, -5.76, width=0.07, color="r")
plt.text(-4.5, 12, "step 1")

plt.xlim(xmin=-5, xmax=5)
plt.ylim(ymin=0, ymax=25)
plt.show()
# -

# The gradient at our new point is still not close to zero, meaning we haven't reached the minimum. So we will take another step:

# +
t = np.linspace(-5, 5, 100)
plt.plot(t, t ** 2)


plt.scatter(-4, 16)  # point

plt.scatter(-3.2, 10.24)  # second point
plt.arrow(-4, 16, 0.8, -5.76, width=0.07, color="r")
plt.text(-4.5, 12, "step 1")

plt.scatter(-2.56, 6.5536)  # third point
plt.arrow(-3.2, 10.24, 0.64, -3.6864, width=0.07, color="r")
plt.text(-4, 7, "step 2")


plt.xlim(xmin=-5, xmax=5)
plt.ylim(ymin=0, ymax=25)
plt.show()
# -

# We would continue this process of checking if the gradient is nearly zero, and taking a step in the direction of steepest descent until we reach the bottom of the valley. We'll say we've reached the bottom of the valley when the absolute value of the gradient is <0.1:

# +
t = np.linspace(-5, 5, 100)
plt.plot(t, t ** 2)

x_vals = np.array(
    [
        -4,
        -3.2,
        -2.56,
        -2.048,
        -1.6384,
        -1.31072,
        -1.048576,
        -0.8388608,
        -0.67108864,
        -0.536870912,
        -0.42949673,
        -0.343597384,
        -0.274877907,
        -0.219902326,
        -0.17592186,
        -0.140737488,
        -0.112589991,
        -0.090071993,
        -0.072057594,
        -0.057646075,
        -0.04611686,
    ]
)
y_vals = np.array(
    [
        16,
        10.24,
        6.5536,
        4.194304,
        2.68435456,
        1.717986918,
        1.099511628,
        0.703687442,
        0.450359963,
        0.288230376,
        0.184467441,
        0.118059162,
        0.075557864,
        0.048357033,
        0.030948501,
        0.019807041,
        0.012676506,
        0.008112964,
        0.005192297,
        0.00332307,
        0.002126765,
    ]
)

plt.scatter(x_vals, y_vals)

plt.xlim(xmin=-5, xmax=5)
plt.ylim(ymin=0, ymax=25)
plt.show()
# -

# This works, but it takes a long time since we have several small steps. We could speed up the process by taking large steps. Starting at (-4,16) and increasing the step of size to 1.5 we find:

# +
q = np.linspace(-20, 20, 1000)
plt.plot(q, q ** 2)


x_vals = np.array([-4, 8])
y_vals = np.array([16, 64])

plt.plot(x_vals, y_vals)


plt.xlim(xmin=-20, xmax=20)
plt.ylim(ymin=0, ymax=260)
plt.show()
# -

# *Note the change in scale.* With only one step, we already see that we stepped over the minimum! If we look at the tangent line at this point we see that the direction of the steepest descent is now to the left instead of the right:

# +
q = np.linspace(-20, 20, 1000)
plt.plot(q, q ** 2)


plt.scatter(-4, 16)  # point
qp = -4
yp = 16


plt.scatter(8, 64)  # second point
qp2 = 8
yp2 = 64


plt.plot(q, qp2 * 2 * (q - qp2) + yp2)  # tangent line

plt.xlim(xmin=-20, xmax=20)
plt.ylim(ymin=0, ymax=260)
plt.show()
# -

# In our attempt to continue to find the minimum, we would now take a step of the same size, but now to the left. This puts us back at on the left side of the hill, but now above where we started. We would be stuck going back and forth between the two sides of the hill continuing to go upward.

# We know as we get closer to the minimum the slope/gradient levels out. This means if we are very far from the minimum, we know we can take a very large step without stepping over the minimum. However, if we are close to the minimum, with a very small slope, we need to be careful and take smaller steps.
#
# We would continue this process of (1) Calculating our slope, (2) Determining a step size, and (3) Moving that step size until we finally reached a spot where the slope or gradient is essentially zero.
#
# This process is a gradient descent and it is used to minimize functions.

# +
q = np.linspace(-20, 20, 1000)
plt.plot(q, q ** 2)


x_vals = np.array([-4, 8, -16])
y_vals = np.array([16, 64, 256])

plt.plot(x_vals, y_vals)


plt.xlim(xmin=-20, xmax=20)
plt.ylim(ymin=0, ymax=260)
plt.show()
# -

#
# This is why it is important to pick the proper step size- also known as the learning rate. Too small steps take a long time while steps that are too large may cause us to miss the minimum. We should pick a step size that is in between the two. For example, in this case a reasonable choice would have been `step size = 0.6`, as it would have approximately reached the minimum after 3 steps.
#
#
# This process of: (1)Calculating the gradient at a point, (2) Determining if the gradient is within the stopping criterion (in this case, the gradient is about equal to zero or <0.01), and (3) Taking a step if the criterion is not met, is known as Gradient Descent.
#
# ## Additional Resources
#
# * [PyTorch documentation on autograd](https://pytorch.org/docs/stable/autograd.html)
# * [Angus Williams' blog post on autodifferentiaton, JAX, and Laplace's method](https://anguswilliams91.github.io/statistics/computing/jax/)
# * [Paperspace blog post on understanding graphs and automatic differentiation](https://blog.paperspace.com/pytorch-101-understanding-graphs-and-automatic-differentiation/)

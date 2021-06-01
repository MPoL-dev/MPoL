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

# Let's define some variable $y$ in terms of $x$:

y = x ** 2

#  We see that the value of $y$ is as we expect---nothing too strange here.

print(f"y: {y}")
print(f"x: {x}")

# But what if we wanted to calculate the gradient of $y$ with respect to $x$? Using calculus, we find that the answer is $\frac{dy}{dx} = 2*x$. The derivative evaluated at $x = 3$ is $6$.
#
# We can see if PyTorch gets the same answer as us if we do:

y.backward()  # populates gradient (.grad) attributes of y with respect to all of its independent variables
x.grad  # returns the grad attribute (the gradient) of y with respect to x

# PyTorch uses the concept of automatic differentiation to calculate the derivative. Instead of computing the derivative as we would by hand, the program is using a computational graph. This is a mechanistic application of the chain rule. For example, a tree with several operations on $x$ resulting in a final output $y$ will use the chain rule to compute the differential associated with each operation and multiply these differentials together to get the derivative of y with respect to x.

# ## Optimizing a Function with Gradient Descent
#
# If we were on the side of a hill in the dark and we wanted to get down to the bottom of a valley, how would we do it?
#
# We wouldn't be able to see all the way to the bottom of the valley, but we could feel which way is down based on where we are standing. We would take steps in the downward direction and we'd know when to stop when the ground felt flat.
#
# One other thing we'd have to consider is our step size. If we take very small steps in the direction of the descent, it will take us a longer time than if we take larger steps. However, if we take super long steps, we might completely miss the flat part of the valley, and start ascending the other side of the valley.

# We can look at the gradient descent from a more mathematical lense by looking at the graph $y = x^2$:

# +
def y(x_input):
    y = x_input ** 2
    return y


x = np.linspace(-5, 5, 100)
plt.plot(x, y(x))  # plot y = x ** 2
# -

# We will choose to start on the left hill at the point (-4, 16). This is an arbitrary choice, any point could've been chosen as the start:

# +
x = np.linspace(-5, 5, 100)
plt.plot(x, y(x))  # plot y = x ** 2

x_start = -4  # x coordinate of starting point
y_start = 16  # y coordinate of starting point

plt.scatter(x_start, y_start)  # plot starting point
plt.show()
# -

# If we plot a tangent line at this point, we see which directions we can go:

# +
x = np.linspace(-5, 5, 100)
plt.plot(x, y(x))  # plot y = x ** 2


plt.scatter(x_start, y_start)  # plot starting point

slope_start = 2 * x_start  # derivative of y = x ** 2 evaluated at x_start
plt.plot(x, slope_start * (x - x_start) + y_start)  # tangent line

plt.xlim(xmin=-5, xmax=5)
plt.ylim(ymin=0, ymax=25)
plt.show()
# -

# We see we need to go down to go toward the minimum. We take a very small step in the direction of the steepest downward slope. When we take steps, we find the x coordinate of our new location by this equation:
#
# $x_{new} = x_{current} - \nabla y(x_{current}) * (step \: size)$
#
# where:
#
# - $x_{current}$ is our current x value
#
# - $\nabla y(x_{current})$ is the gradient at our current point
#
# - $(step \: size)$ is a value we choose that scales our steps
#
# This makes sense because we start at some x value and we want to get to a new x value that is closer to the minimum. In the above plot, the gradient is negative. If we were to add the gradient to our current x value that would bring us further away from the minimum to a more negative x value. This is because the gradient points in the direction of the steepest ascent while we are looking to go in the direction of the steepest descent. This is why there is a negative in the equation.
#
#
# We will choose <code> step_size = 0.1 </code>:

# +
x = np.linspace(-5, 5, 100)
plt.plot(x, y(x))  # plot y = x ** 2

# We chose step size of 0.1
step_size = 0.1

# Define a function to get slope of y = x ** 2 at any x value:
def dy_dx(x_input):
    gradient = 2 * x_input  # derivative of y = x ** 2
    return gradient


# Current values at starting point we chose:
x_current = x_start
y_current = y_start
dy_dx_current = dy_dx(x_current)

# To keep track of our coordinates at each step, we will create 2 lists, initialized with the values at our chosen starting point
x_coords = [x_current]
y_coords = [y_current]


# Using equation for x_new to get second point
x_new = x_current - dy_dx(x_current) * step_size

# Plug in x_new into y = x ** 2 to get y_new of second point
y_new = y(x_new)


# Store second point coordinates in our lists
x_coords.append(x_new)
y_coords.append(y_new)


plt.scatter(x_coords, y_coords)  # plot points showing steps
plt.text(-4.5, 12, "step 1")

plt.xlim(xmin=-5, xmax=5)
plt.ylim(ymin=0, ymax=25)
plt.show()
# -

# The gradient at our new point is still not close to zero, meaning we haven't reached the minimum. We continue this process of checking if the gradient is nearly zero, and taking a step in the direction of steepest descent until we reach the bottom of the valley. We'll say we've reached the bottom of the valley when the absolute value of the gradient is $<0.1$:

# +
x = np.linspace(-5, 5, 100)
plt.plot(x, y(x))  # plot y = x ** 2

# We are now at our second point so we need to update our current values
x_current = x_new
y_current = y_new


# We automate this process with the following while loop
while abs(dy_dx(x_current)) >= 0.1:  # Check to see if we're at minimum
    # Get new coordinates
    x_new = x_current - dy_dx(x_current) * step_size
    y_new = y(x_new)

    # Add new coordinates to lists
    x_coords.append(x_new)
    y_coords.append(y_new)

    # Update current position
    x_current = x_new


plt.scatter(x_coords, y_coords)  # plot points showing steps

plt.xlim(xmin=-5, xmax=5)
plt.ylim(ymin=0, ymax=25)
plt.show()
# -

# This works, but it takes a long time since we have several small steps. We could speed up the process by taking large steps.  We're only focused on the affects of changing the step size, so we will keep (-4, 16) as the starting point and increase the step size to $1.5$. Our first step now looks like:

# +
x_large_step = np.linspace(-20, 20, 1000)
plt.plot(x_large_step, y(x_large_step))  # plot y = x ** 2

# Current values at starting point we chose:
x_large_step_current = x_start
y_large_step_current = y_start
dy_dx_current = dy_dx(x_large_step_current)

# To keep track of our coordinates at each step, we will create 2 lists, initialized with the values at our chosen starting point
x_large_coords = [x_large_step_current]
y_large_coords = [y_large_step_current]

# New step_size
large_step_size = 1.5

# Get coordinates of our second point using x_new equation and y = x ** 2
x_large_step_new = x_large_step_current - dy_dx(x_large_step_current) * large_step_size
y_large_step_new = y(x_large_step_new)

# Store second point coordinates in our lists
x_large_coords.append(x_large_step_new)
y_large_coords.append(y_large_step_new)


plt.plot(x_large_coords, y_large_coords)  # plot points showing steps


plt.xlim(xmin=-20, xmax=20)
plt.ylim(ymin=0, ymax=260)
plt.show()
# -

# *Note the change in scale.* With only one step, we already see that we stepped over the minimum! If we look at the tangent line at this point we see that the direction of the steepest descent is now to the left instead of the right:

# +
x_large_step = np.linspace(-20, 20, 1000)
plt.plot(x_large_step, y(x_large_step))  # plot y = x ** 2


plt.plot(x_large_coords, y_large_coords)  # plot points showing steps


plt.plot(
    x_large_step,
    dy_dx(x_large_step_new) * (x_large_step - x_large_step_new) + y_large_step_new,
)  # tangent line

plt.xlim(xmin=-20, xmax=20)
plt.ylim(ymin=0, ymax=260)
plt.show()
# -

# In our attempt to continue to find the minimum, we would take a step of the same size, but now to the left. This puts us back on the left side of the hill, but now above where we started. We would be stuck going back and forth between the two sides of the hill continuing to go upward.

# +
x_large_step = np.linspace(-20, 20, 1000)
plt.plot(x_large_step, y(x_large_step))  # plot y = x ** 2

# We are now at our second point so we need to update our current values
x_large_step_current = x_large_step_new
y_large_step_current = y_large_step_new


# We automate this process with the following while loop

num_iter = 0  # To keep track of number of steps
while abs(dy_dx(x_large_step_current)) > 0.1:  # Check to see if we're at minimum
    # Get new coordinates
    x_large_step_new = (
        x_large_step_current - dy_dx(x_large_step_current) * large_step_size
    )
    y_large_step_new = y(x_large_step_new)

    # Add new coordinates to lists
    x_large_coords.append(x_large_step_new)
    y_large_coords.append(y_large_step_new)

    # Update current position
    x_large_step_current = x_large_step_new

    # Update number of iterations
    num_iter = num_iter + 1

    # Break loop if minimum not found within 20 steps
    if num_iter > 20:
        break


plt.plot(x_large_coords, y_large_coords)  # plot points showing steps


plt.xlim(xmin=-20, xmax=20)
plt.ylim(ymin=0, ymax=260)
plt.show()
# -

#
# This is why it is important to pick the proper step size- also known as the learning rate. Steps that are too small take a long time while steps that are too large may cause us to miss the minimum. We should pick a step size that is in between the two. For example, in this case a reasonable choice would have been <code> step size = 0.6 </code>, as it would have approximately reached the minimum after 3 steps.
#
#
# This process of:
#
# * Calculating the gradient at a point
# * Determining if the gradient is within the stopping criterion (in this case, the gradient is about equal to zero or $<0.1$)
# * Taking a step if the criterion is not met
#
#  is known as Gradient Descent.
#
# ## Additional Resources
#
# * [PyTorch documentation on autograd](https://pytorch.org/docs/stable/autograd.html)
# * [Angus Williams' blog post on autodifferentiaton, JAX, and Laplace's method](https://anguswilliams91.github.io/statistics/computing/jax/)
# * [Paperspace blog post on understanding graphs and automatic differentiation](https://blog.paperspace.com/pytorch-101-understanding-graphs-and-automatic-differentiation/)

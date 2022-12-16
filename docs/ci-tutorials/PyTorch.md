---
jupytext:
  encoding: '# -*- coding: utf-8 -*-'
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

```{code-cell}
:tags: [hide-cell]
%run notebook_setup
```

# Introduction to PyTorch: Tensors and Gradient Descent

This tutorial provides a gentle introduction to PyTorch tensors, automatic differentiation, and optimization with gradient descent outside of any specifics about radio interferometry or the MPoL package itself.

## Introduction to Tensors

Tensors are multi-dimensional arrays, similar to numpy arrays, with the added benefit that they can be used to calculate gradients (more on that later). MPoL is built on the [PyTorch](https://pytorch.org/) machine learning library, and uses a form of gradient descent optimization to find the "best" image given some dataset and loss function, which may include regularizers.

We'll start this tutorial by importing the torch and numpy packages. Make sure you have [PyTorch installed](https://pytorch.org/get-started/locally/) before proceeding.

```{code-cell}
import matplotlib.pyplot as plt
import numpy as np
import torch
```

There are several [ways to initialize a tensor](https://pytorch.org/docs/stable/tensors.html). A common method to create a tensor is from a numpy array:

```{code-cell}
an_array = np.array([[1, 2], [3, 4]])
a_tensor = torch.tensor(an_array)  # creates tensor of same size as an_array

print(a_tensor)
```

Tensors are similar to numpy arrays---many of the same [operations](https://pytorch.org/docs/stable/torch.html) that we would perform on numpy arrays can easily be performed on PyTorch tensors. For example, we can compare how to calculate a matrix product using numpy and PyTorch

```{code-cell}
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
```

## Calculating Gradients

+++

PyTorch allows us to calculate the gradients on tensors, which is a key functionality underlying MPoL. Let's start by creating a tensor with a single value. Here we are setting ``requires_grad = True``; we'll see why this is important in a moment.

```{code-cell}
x = torch.tensor(3.0, requires_grad=True)
x
```

Let's define some variable $y$ in terms of $x$:

```{code-cell}
y = x ** 2
```

We see that the value of $y$ is as we expect---nothing too strange here.

```{code-cell}
print(f"x: {x}")
print(f"y: {y}")
```

But what if we wanted to calculate the gradient of $y$ with respect to $x$? Using calculus, we find that the answer is $\frac{dy}{dx} = 2x$. The derivative evaluated at $x = 3$ is $6$.

We can use PyTorch to get the same answer---no analytic derivative needed!

```{code-cell}
y.backward()  # populates gradient (.grad) attributes of y with respect to all of its independent variables
x.grad  # returns the grad attribute (the gradient) of y with respect to x
```

PyTorch uses the concept of [automatic differentiation](https://arxiv.org/abs/1502.05767) to calculate the derivative. Instead of computing the derivative as we would by hand, the program uses a computational graph and the mechanistic application of the chain rule. For example, a computational graph with several operations on $x$ resulting in a final output $y$ will use the chain rule to compute the differential associated with each operation and multiply these differentials together to get the derivative of $y$ with respect to $x$.

+++

## Optimizing a Function with Gradient Descent

If we were on the side of a hill in the dark and we wanted to get down to the bottom of a valley, how might we do it?

We can't see all the way to the bottom of the valley, but we can feel which way is down based on the incline of where we are standing. We might take steps in the downward direction and we'd know when to stop when the ground finally felt flat. We would also need to consider how large our steps should be. If we take very small steps, it will take us a longer time than if we take larger steps. However, if we take large leaps, we might completely miss the flat part of the valley, and jump straight across to the other side of the valley.

Now let's take a more quantitative look at the gradient descent using the function $y = x^2$:

```{code-cell}
def y(x):
    return torch.square(x)
```

We will choose some arbitrary place to start on the left side of the hill and use PyTorch to calculate the tangent.

Note that the plotting library Matplotlib requires numpy arrays instead of PyTorch tensors, so in the following code you might see the occasional ``detach().numpy()`` or ``.item()`` calls, which are used to convert PyTorch tensors to numpy arrays and scalar values, respectively, for plotting. When it comes time to use MPoL for RML imaging, or any large production run, we'll try to keep the calculations native to PyTorch tensors as long as possible, to avoid the overhead of converting types.

```{code-cell}
x = torch.linspace(-5, 5, 100)
plt.plot(x, y(x))

x_start = torch.tensor(
    -4.0, requires_grad=True
)  # tensor with x coordinate of starting point
y_start = y(x_start)  # tensor with y coordinate of starting point

plt.scatter(x_start.item(), y_start.item())  # plot starting point

# we can calculate the derivative of y = x ** 2 evaluated at x_start
y_start.backward()  # populate x_start.grad
slope_start = x_start.grad

# and use this to evaluate the tangent line
tangent_line = slope_start * (x - x_start) + y_start

plt.plot(x, tangent_line.detach().numpy())
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.xlim(xmin=-5, xmax=5)
plt.ylim(ymin=0, ymax=25)
plt.show()
```

We see we need to go to the right to go down toward the minimum. For a multivariate function, the gradient will be a vector pointing in the direction of the steepest downward slope. When we take steps, we find the x coordinate of our new location by:

$x_\mathrm{new} = x_\mathrm{current} - \nabla y(x_\mathrm{current}) * (\mathrm{step\,size})$

where:

- $x_\mathrm{current}$ is our current x value
- $\nabla y(x_\mathrm{current})$ is the gradient at our current point
- $(\mathrm{step\,size})$ is a value we choose that scales our steps

We will choose ``step_size = 0.1``:

```{code-cell}
x = torch.linspace(-5, 5, 100)
plt.plot(x, y(x), zorder=0)  # plot y = x ** 2

step_size = 0.1

# Tensors containing current coordinates at the starting point we chose:
x_current = x_start
y_current = y(x_current)

# To keep track of our coordinates at each step, we will create 2 lists, initialized with the values at our chosen starting point
# These lists will be used to plot points with Matplotlib.pyplot so we use .item() to only retain the value in the tensor
x_coords = [x_current.item()]
y_coords = [y_current.item()]

# Slope at current point
y_current.backward()  # populate x_current.grad
slope_current = (
    x_current.grad
)  # tensor containing derivative of y = x ** 2 evaluated at current point

# Using equation for x_new to get x coordinate of second point, store it in a tensor
# We cannot use torch.tensor(...) to make a new tensor from previous tensors without altering the
# computational graph. We use .item() to only use float values to create our new tensor
x_new = torch.tensor(
    x_current.item() - (slope_current.item()) * step_size, requires_grad=True
)

# Plug in x_new into y = x ** 2 to get y_new of second point
y_new = y(x_new)


# Store second point coordinates in our lists
x_coords.append(x_new.item())
y_coords.append(y_new.item())


plt.scatter(x_coords, y_coords)  # plot points showing steps
# replot the last point in a new color
plt.scatter(x_coords[-1], y_coords[-1], c="C1", zorder=1)
plt.text(-2, 5, "step 1", va="center")

plt.xlim(xmin=-5, xmax=5)
plt.ylim(ymin=-1, ymax=25)
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.show()
```

The gradient at our new point (shown in orange) is still not close to zero, meaning we haven't reached the minimum. We'll continue this process of checking if the gradient is nearly zero, and take a step in the direction of steepest descent until we reach the bottom of the valley. We'll say we've reached the bottom of the valley when the absolute value of the gradient is $<0.1$:

```{code-cell}
x = torch.linspace(-5, 5, 100)
plt.plot(x, y(x), zorder=0)  # plot y = x ** 2

# We are now at our second point so we need to update our tensors containing our current coordinates
x_current = x_new
y_current = y_new


# We automate this process with the following while loop
y_current.backward()  # populate x_current.grad
while abs(x_current.grad) >= 0.1:  # Check to see if we're at minimum
    # Get tensors containing new coordinates
    x_new = torch.tensor(
        x_current.item() - x_current.grad.item() * step_size, requires_grad=True
    )
    y_new = y(x_new)

    # Add new coordinates to lists
    x_coords.append(x_new.item())
    y_coords.append(y_new.item())

    # Update current position
    x_current = x_new
    y_current = y_new

    # Update current slope
    y_current.backward()  # populate x_current.grad


plt.scatter(x_coords, y_coords)  # plot points showing steps
plt.scatter(x_coords[-1], y_coords[-1], c="C1")  # highlight last point

plt.xlim(xmin=-5, xmax=5)
plt.ylim(ymin=-1, ymax=25)
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.show()
```

This works, but it takes a long time since we have several small steps.

Can we speed up the process by taking large steps? Most likely, yes. But there is a danger in taking step sizes that are too large. For example, let's repeat this exercise with a step size of $1.5$. Our first step now looks like:

```{code-cell}
x_large_step = torch.linspace(-20, 20, 1000)
plt.plot(x_large_step, y(x_large_step), zorder=0)  # plot y = x ** 2

# Current values at starting point we chose:
x_large_step_current = torch.tensor(-4.0, requires_grad=True)
y_large_step_current = y(x_large_step_current)

# Slope at current point
y_large_step_current.backward()  # populate x_large_step_current.grad
slope_large_step_current = (
    x_large_step_current.grad
)  # tensor containing derivative of y = x ** 2 evaluated at current point

# To keep track of our coordinates at each step, we will create 2 lists, initialized with the coordinates at our chosen starting point
# These lists will be used to plot points with Matplotlib.pyplot so we use .item() to only retain the value in the tensor
x_large_coords = [x_large_step_current.item()]
y_large_coords = [y_large_step_current.item()]

# New step_size
large_step_size = 1.5

# Get coordinates of our second point using x_new equation and y = x ** 2
x_large_step_new = torch.tensor(
    x_large_step_current.item() - slope_large_step_current.item() * large_step_size,
    requires_grad=True,
)
y_large_step_new = y(x_large_step_new)

# Store second point coordinates in our lists
x_large_coords.append(x_large_step_new.item())
y_large_coords.append(y_large_step_new.item())


plt.scatter(x_large_coords, y_large_coords)  # plot points showing steps
plt.scatter(x_large_coords[-1], y_large_coords[-1], c="C1")
plt.text(9, 70, "step 1", va="center")

plt.xlim(xmin=-20, xmax=20)
plt.ylim(ymin=-1, ymax=260)
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.show()
```

*Note the change in scale!* With only one step, we already see that we stepped *right over* the minimum to somewhere far up the other side of the valley (orange point)! This is not good. If we kept iterating with the same learning rate, we'd find that the optimization process diverges and the step sizes start blowing up. This is why it is important to pick the proper step size by setting the learning rate appropriately. Steps that are too small take a long time while steps that are too large render the optimization process invalid. In this case, a reasonable choice appears to be ``step size = 0.6``, which would have reached pretty close to the minimum after only 3 steps.

To sum up, optimizing a function with gradient descent consists of

1. Calculate the gradient at your current point
2. Determine if the gradient is within the stopping criterion (in this case, the gradient is about equal to zero or $<0.1$), if so stop
3. Otherwise, take a step in the direction of the gradient and go to #1

Autodifferentiation frameworks like PyTorch allow us to easily calculate the gradient of complex functions, including a large set of prior/regularizer functions that we would want to use for Regularized Maximum Likelihood (RML) imaging. This makes it relatively easy to quickly and efficiently solve for the "optimal" image given a set of data and regularizer terms.

## Additional Resources

* [PyTorch documentation on autograd](https://pytorch.org/docs/stable/autograd.html)
* [Angus Williams' blog post on autodifferentiaton, JAX, and Laplace's method](https://anguswilliams91.github.io/statistics/computing/jax/)
* [Paperspace blog post on understanding graphs and automatic differentiation](https://blog.paperspace.com/pytorch-101-understanding-graphs-and-automatic-differentiation/)
* [3Blue1Brown video on gradient descent](https://youtu.be/IHZwWFHWa-w)

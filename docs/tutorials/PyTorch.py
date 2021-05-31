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
# Another thing to note, if there were multiple paths connecting a leaf to a tensor, the grad attribute would be calculated by (1) Multiplying the partial derivatives along each edge of a single path together (2) Adding up the products of each path together.  To see an example of this go [here](https://blog.paperspace.com/pytorch-101-understanding-graphs-and-automatic-differentiation/).
#

# ## Idea of Gradient Descent
#
# If we were on the side of a hill in the dark and we wanted to get down to the bottom of a valley, how would we do it?
#
# We wouldn't be able to see all the way to the bottom of the valley, but we could feel which way is down based on where we are standing. We would take steps in the downward direction and we'd know when to stop when the ground felt flat.
#
# One other thing we'd have to consider is our step size. If we take very small steps in the direction of the descent, it will take us a longer time than if we take larger steps. However, if we take super long steps, we might completely miss the flat part of the valley, and start ascending the other side of the valley.

# ## Thinking about Gradient Descent Mathematically

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
# - $\nabla y(x_{current})$ is the gradient at our current point
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
# This process of
#
# - Calculating the gradient at a point
# - Determining if the gradient is within the stopping criterion (in this case, the gradient is about equal to zero or <0.01)
# - Taking a step if the criterion is not met
#
# is known as Gradient Descent

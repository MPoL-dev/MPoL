# Introduction and Orientation

2 pages max.

first par
* restate MPoL as a library w/ maintenance goal
* highlight science cases: RML imaging, Bayesian inference
* discuss scope of documentation and need for pre-requisites

second par
* brief description to what RML is, with call-outs to Zawadzki and other seminal resources.

third par
* discuss pre-requisites: radio background, 
* pytorch background

fourth par
* scope of tutorial examples to follow, showing module building blocks
* browse the API
* organization of examples folder




(rml-intro-label)=

# Introduction to Regularized Maximum Likelihood Imaging

This document is an attempt to provide a whirlwind introduction to what Regularized Maximum Likelihood (RML) imaging is, and why you might want to use this MPoL package to perform it with your interferometric dataset. Of course, the field is rich, varied, and this short introduction couldn't possibly do justice to cover the topic in depth. We recommend that you check out many of the links and suggestions in this document for further reading and understanding.



## The MPoL package for Regularized Maximum Likelihood imaging

*Million Points of Light* or "MPoL" is a Python package that is used to perform regularized maximum likelihood imaging. By that we mean that the package provides the building blocks to create flexible image models and optimize them to fit interferometric datasets. The package is developed completely in the open on [Github](https://github.com/MPoL-dev/MPoL).

We strive to

- create an open, welcoming, and supportive community for new users and contributors (see our [code of conduct](https://github.com/MPoL-dev/MPoL/blob/main/CODE_OF_CONDUCT.md) and [developer documentation](developer-documentation.md))
- support well-tested and stable releases (i.e., `pip install mpol`) that run on all currently-supported Python versions, on Linux, MacOS, and Windows
- maintain up-to-date {ref}`API documentation <api-reference-label>`
- cultivate tutorials covering real-world applications

:::{seealso}
We also recommend checking out several other excellent packages for RML imaging:

- [SMILI](https://github.com/astrosmili/smili)
- [eht-imaging](https://github.com/achael/eht-imaging)
- [GPUVMEM](https://github.com/miguelcarcamov/gpuvmem)
:::

There are a few things about  MPoL that we believe make it an appealing platform for RML modeling.

**Built on PyTorch**: Many of MPoL's exciting features stem from the fact that it is built on top of a rich computational library that supports autodifferentiation and construction of complex neural networks. Autodifferentiation libraries like [Theano/Aesara](https://github.com/aesara-devs/aesara), [Tensorflow](https://www.tensorflow.org/), [PyTorch](https://pytorch.org/), and [JAX](https://jax.readthedocs.io/) have revolutionized the way we compute and optimize functions. For now, PyTorch is the library that best satisfies our needs, but we're keeping a close eye on the Python autodifferentiation ecosystem should a more suitable framework arrive. If you are familiar with scientific computing with Python but haven't yet tried any of these frameworks, don't worry, the syntax is easy to pick up and quite similar to working with numpy arrays. For example, check out our tutorial [introduction to PyTorch](ci-tutorials/PyTorch.md).

**Autodifferentiation**: PyTorch gives MPoL the capacity to autodifferentiate through a model. The *gradient* of the objective function is exceptionally useful for finding the "downhill" direction in a large parameter space (such as the set of image pixels). Traditionally, these gradients would have needed to been calculated analytically (by hand) or via finite-difference methods which can be noisy in high dimensions. By leveraging the autodifferentiation capabilities, this allows us to rapidly formulate and implement complex prior distributions which would otherwise be difficult to differentiate by hand.

**Optimization**: PyTorch provides a full-featured suite of research-grade [optimizers](https://pytorch.org/docs/stable/optim.html) designed to train deep neural networks. These same optimizers can be employed to quickly find the optimum RML image.

**GPU acceleration**: PyTorch wraps CUDA libraries, making it seamless to take advantage of (multi-)GPU acceleration to optimize images. No need to use a single line of CUDA.

**Model composability**: Rather than being a monolithic program for single-click RML imaging, MPoL strives to be a flexible, composable, RML imaging *library* that provides primitives that can be used to easily solve your particular imaging challenge. One way we do this is by mimicking the PyTorch ecosystem and writing the RML imaging workflow using [PyTorch modules](https://pytorch.org/tutorials/beginner/nn_tutorial.html). This makes it easy to mix and match modules to construct arbitrarily complex imaging workflows. We're working on tutorials that describe these ideas in depth, but one example would be the ability to use a single latent space image model to simultaneously fit single dish and interferometric data.

**A bridge to the machine learning/neural network community**: MPoL will happily calculate RML images for you using "traditional" image priors, lest you are the kind of person that turns your nose up at the words "machine learning" or "neural network." However, if you are the kind of person that sees opportunity in these tools, because MPoL is built on PyTorch, it is straightforward to take advantage of them for RML imaging. For example, if one were to train a variational autoencoder on protoplanetary disk emission morphologies, the latent space + decoder architecture could be easily plugged in to MPoL and serve as an imaging basis set.

To get started with MPoL, we recommend [installing the package](installation.md) and reading through the tutorial series. If you have any questions about the package, we invite you to join us on our [Github discussions page](https://github.com/MPoL-dev/MPoL/discussions).

:::{seealso}
That's RML imaging in a nutshell, but we've barely scratched the surface. We highly recommend checking out the following excellent resources.

- The fourth paper in the 2019 [Event Horizon Telescope Collaboration series](https://ui.adsabs.harvard.edu/abs/2019ApJ...875L...4E/abstract) describing the imaging principles
- [Maximum entropy image restoration in astronomy](https://ui.adsabs.harvard.edu/abs/1986ARA%26A..24..127N/abstract) AR&A by Narayan and Nityananda 1986
- [Multi-GPU maximum entropy image synthesis for radio astronomy](https://ui.adsabs.harvard.edu/abs/2018A%26C....22...16C/abstract) by Cárcamo et al. 2018
:::



```{admonition} CLEAN
RML imaging is different from CLEAN imaging, which operates as a deconvolution procedure in the image plane. CLEAN is by far the dominant algorithm used to synthesize images from interferometric data at sub-mm and radio wavelengths. 

- [Interferometry and Synthesis in Radio Astronomy](https://ui.adsabs.harvard.edu/abs/2017isra.book.....T/abstract) Chapter 11.1
- [CASA documentation on tclean](https://casa.nrao.edu/casadocs-devel/stable/imaging/synthesis-imaging)
- David Wilner's lecture on [Imaging and Deconvolution in Radio Astronomy](https://www.youtube.com/watch?v=mRUZ9eckHZg)
- For a discussion on using both CLEAN and RML techniques to robustly interpret kinematic data of protoplanetary disks, see Section 3 of [Visualizing the Kinematics of Planet Formation](https://ui.adsabs.harvard.edu/abs/2020arXiv200904345D/abstract) by The Disk Dynamics Collaboration
```


```{rubric} Footnotes
```

[^mle-solution]: There's actually a lot to unpack here. When your model has many parameters (i.e., the posterior distribution is high dimensional), the MLE (or MAP) solution is unlikely to represent a *typical* realization of your model parameters. This is a quirk of the geometry of high dimensional spaces. For more information, we recommend checking out Chapter 1 of [Betancourt 2017](https://arxiv.org/abs/1701.02434). Still, the MLE solution is often a useful quantity to communicate, summarizing the mode of the probability distribution.

[^relative-strength]: This is where the factor of $1/2$ in front of $\chi^2$ becomes important. You could use something like $L_\mathrm{nll}(\boldsymbol{\theta}) = \chi^2(\boldsymbol{\theta})$, but then you'd need to change the value of $\lambda_\mathrm{sparsity}$ to achieve the same relative regularization.

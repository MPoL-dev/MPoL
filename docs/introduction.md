# Orientation

*Million Points of Light* or "MPoL" is a [PyTorch](https://pytorch.org/) *library* supporting Regularized Maximum Likelihood (RML) imaging and Bayesian Inference workflows with Fourier datasets. We provide the building blocks to create flexible image models and optimize them to fit interferometric datasets.

## Background and prerequisites

### Radio astronomy

A background in radio astronomy, Fourier transforms, and interferometry is a prerequisite for using MPoL but is beyond the scope of this documentation. We recommend reviewing these resources as needed.

- [Essential radio astronomy](https://www.cv.nrao.edu/~sransom/web/xxx.html) textbook by James Condon and Scott Ransom, and in particular, Chapter 3.7 on Radio Interferometry.
- NRAO's [17th Synthesis Imaging Workshop](http://www.cvent.com/events/virtual-17th-synthesis-imaging-workshop/agenda-0d59eb6cd1474978bce811194b2ff961.aspx) recorded lectures and slides available
- [Interferometry and Synthesis in Radio Astronomy](https://ui.adsabs.harvard.edu/abs/2017isra.book.....T/abstract) by Thompson, Moran, and Swenson. An excellent and comprehensive reference on all things interferometry.
- Ian Czekala's lecture notes on [Radio Interferometry and Imaging](https://iancze.github.io/courses/as5003/lectures/)

RML imaging is different from CLEAN imaging, which operates as a deconvolution procedure in the image plane. However, CLEAN is by far the dominant algorithm used to synthesize images from interferometric data at sub-mm and radio wavelengths, and it is useful to have at least a basic understanding of how it works. We recommend

- [Interferometry and Synthesis in Radio Astronomy](https://ui.adsabs.harvard.edu/abs/2017isra.book.....T/abstract) Chapter 11.1
- David Wilner's lecture on [Imaging and Deconvolution in Radio Astronomy](https://www.youtube.com/watch?v=mRUZ9eckHZg)
- For a discussion on using both CLEAN and RML techniques to robustly interpret kinematic data of protoplanetary disks, see Section 3 of [Visualizing the Kinematics of Planet Formation](https://ui.adsabs.harvard.edu/abs/2020arXiv200904345D/abstract) by The Disk Dynamics Collaboration

### Statistics and Machine Learning

MPoL is built on top of the [PyTorch](https://pytorch.org/) machine learning framework and adopts much of the terminology and design principles of machine learning workflows. As a prerequisite, we recommend at least a basic understanding of statistics and machine learning principles. Two excellent (free) textbooks are

- [Dive into Deep Learning](https://d2l.ai/), in particular chapters 1 - 3 to cover the basics of forward models, automatic differentiation, and optimization.
- [Deep Learning: Foundations and Concepts](https://www.bishopbook.com/) for a lengthier discussion of these concepts and other foundational statistical concepts.

And we highly recommend the informative and entertaining 3b1b lectures on [deep learning](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&ab_channel=3Blue1Brown).

### PyTorch

As a PyTorch library, MPoL expects that the user will write Python code that uses MPoL primitives as building blocks to solve their interferometric imaging workflow, much the same way the artificial intelligence community writes Python code that uses PyTorch layers to implement new neural network architectures (for [example](https://github.com/pytorch/examples)). You will find MPoL easiest to use if you follow PyTorch customs and idioms, e.g., feed-forward neural networks, data storage, GPU acceleration, and train/test optimization loops. Therefore, a basic familiarity with PyTorch is considered a prerequisite for MPoL.

If you are new to PyTorch, we recommend starting with the official [Learn the Basics](https://pytorch.org/tutorials/beginner/basics/intro.html) guide. You can also find high quality introductions on YouTube and in textbooks.

## MPoL for Regularized Maximum Likelihood imaging

This document is an attempt to provide a whirlwind introduction to what Regularized Maximum Likelihood (RML) imaging is, and why you might want to use this MPoL package to perform it with your interferometric dataset. Of course, the field is rich, varied, and this short introduction couldn't possibly do justice to cover the topic in depth. We recommend that you check out many of the links and suggestions in this document for further reading and understanding.



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

second par
* brief description to what RML is, with call-outs to Zawadzki and other seminal resources.


## Scope of tutorials nad exmaples

* scope of tutorial examples to follow, showing module building blocks
* browse the API
* organization of examples folder









Longer workflow exmaples are in examples.

Including Pyro.



:::{seealso}
That's RML imaging in a nutshell, but we've barely scratched the surface. We highly recommend checking out the following excellent resources.

- The fourth paper in the 2019 [Event Horizon Telescope Collaboration series](https://ui.adsabs.harvard.edu/abs/2019ApJ...875L...4E/abstract) describing the imaging principles
- [Maximum entropy image restoration in astronomy](https://ui.adsabs.harvard.edu/abs/1986ARA%26A..24..127N/abstract) AR&A by Narayan and Nityananda 1986
- [Multi-GPU maximum entropy image synthesis for radio astronomy](https://ui.adsabs.harvard.edu/abs/2018A%26C....22...16C/abstract) by Cárcamo et al. 2018
:::





```{rubric} Footnotes
```

[^mle-solution]: There's actually a lot to unpack here. When your model has many parameters (i.e., the posterior distribution is high dimensional), the MLE (or MAP) solution is unlikely to represent a *typical* realization of your model parameters. This is a quirk of the geometry of high dimensional spaces. For more information, we recommend checking out Chapter 1 of [Betancourt 2017](https://arxiv.org/abs/1701.02434). Still, the MLE solution is often a useful quantity to communicate, summarizing the mode of the probability distribution.

[^relative-strength]: This is where the factor of $1/2$ in front of $\chi^2$ becomes important. You could use something like $L_\mathrm{nll}(\boldsymbol{\theta}) = \chi^2(\boldsymbol{\theta})$, but then you'd need to change the value of $\lambda_\mathrm{sparsity}$ to achieve the same relative regularization.

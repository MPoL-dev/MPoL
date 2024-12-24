---
title: 'Million Points of Light (MPoL): a PyTorch library for radio interferometric imaging and inference'
tags:
  - Python
  - astronomy
  - imaging
  - fourier
  - radio astronomy
  - radio interferometry
  - machine learning
  - neural networks
authors:
  - name: Ian Czekala
    orcid: 0000-0002-1483-8811
    corresponding: true
    affiliation: 1
  - name: co-authors
  # - name: Jeff Jennings
  #   orcid: 0000-0002-7032-2350
  #   affiliation: 2
  # - name: Brianna Zawadzki
  #   orcid: 0000-0001-9319-1296
  #   affiliation: 3
  # - name: Kadri Nizam
  #   orcid: 0000-0002-7217-446X
  #   affiliation: 2
  # - name: Ryan Loomis
  #   orcid: 0000-0002-8932-1219
  #   affiliation: 4
  # - name: Megan Delamer
  #   orcid: 0000-0003-1439-2781
  #   affiliation: 2     
  # - name: Kaylee de Soto
  #   orcid: 0000-0002-9886-2834
  #   affiliation: 2
  # - name: Robert Frazier
  #   orcid: 0000-0001-6569-3731
  #   affiliation: 2      
  # - name: Hannah Grzybowski
  #   # orcid: # can't find
  #   affiliation: 2         
  # - name: Mary Ogborn
  #   orcid: 0000-0001-9741-2703
  #   affiliation: 2
  # - name: Tyler Quinn
  #   orcid: 0000-0002-8974-8095
  #   affiliation: 2
affiliations:
 - name: University of St Andrews, Scotland 
   index: 1
#  - name: Pennsylvania State University, USA
#    index: 2
#  - name: Wesleyan University, USA
#    index: 3
#  - name: National Radio Astronomy Observatory, Charlottesville, VA, USA
#    index: 4      
date: 16 December 2024
bibliography: paper.bib
aas-journal: Astrophysical Journal
---

# Summary

Astronomical radio interferometers achieve exquisite angular resolution by cross-correlating signal from a cosmic source simultaneously observed by distant pairs of radio telescopes to produce a Fourier-type measurement called a visibility. *Million Points of Light* (`MPoL`) is a Python library supporting feed-forward modeling of interferometric visibility datasets for synthesis imaging and parametric Bayesian inference, built using the autodifferentiable machine learning framework PyTorch. Neural network components provide a rich set of modular and composable building blocks that can be used to express the physical relationships between latent model parameters and observed data following the radio interferometric measurement equation. Industry-grade optimizers make it straightforward to simultaneously solve for the synthesized image and calibration parameters using stochastic gradient descent.

# Statement of need

When an astrophysical source is observed by a radio interferometer, it is typical for there to be gaps in the spatial frequency coverage. Therefore, rather than a direct Fourier inversion, images must be synthesized from the visibility data using an imaging algorithm; it is common for the incomplete sampling to severely hamper image reconstruction. CLEAN is the traditional image synthesis algorithm of the radio interferometry community [@hogbom74; see @thompson17, Ch. 11, for a review], with a modern implementation in the facility software CASA [@mcmullin07; @casa22]. CLEAN excels at the rapid imaging of astronomical fields comprising unresolved point sources (e.g. quasars) and marginally resolved sources, but often struggles to achieve the desired imaging performance for spatially extended sources [@disk20, §3].

In the field of planet formation alone, spatially resolved observations from the Atacama Large Millimeter Array (ALMA) have rapidly advanced our understanding of protoplanetary disk structures [@andrews20], kinematic signatures of embedded protoplanets [@pinte18], and circumplanetary disks [@benisty21]. High fidelity imaging algorithms for spatially resolved sources are needed to realize the full scientific potential of groundbreaking observatories like ALMA [@wootten09], the Event Horizon Telescope [@eht19a], and the Square Kilometer Array [@dewdney09] as they deliver significantly improved sensitivity and resolving power compared to previous generation instruments. Moreover, there is an opportunity for a flexible, open-source platform to interface with advanced algorithms provided by machine learning and computational imaging software from non-astronomy fields.

# The Million Points of Light (MPoL) library

<!-- forward modeling library philosophy -->
`MPoL` is a library designed for the feed-forward modeling of interferometric datasets using Python, Numpy [@harris20], and the computationally performant machine learning framework PyTorch [@paszke19]. `MPoL` implements a set of primatives and foundational components using PyTorch `nn.module` which can be easily arranged to meet the requirements of the forward modeling problem at hand. Rather than provide a standalone, end-to-end imaging solution, `MPoL`'s philosophy is to closely integrate with the PyTorch ecosystem so that users can leverage powerful PyTorch idioms for machine learning workflows. Because of this design decision, the user can transparently tap into the rich PyTorch ecosystem: GPU acceleration, parametric inference, and other trained neural network models.

<!-- high-level walk-through of what forward modeling is mentioning the components-->
Thus far, MPoL has focused on workflows related to the synthesis of images of astrophysical sources using continuum, spectral line data. Thus far, our group has worked with protoplanetary disks, since that's our speciality, but code is more general to everything. These work using Regularized Maximum Likelihood (RML) principles. Primatives for spherical coordinates. In a feed-forward sense, Foundational components for Pixel basis / image layer, Fourier layers to transform, which can either be gridded or  Interpolation to individual baselines possible using NuFFT using kbnufft [@nufft20]. Then, comparison to data using common loss functions. However, because of the aforementioned sparsity of spatial frequency samples, there are a number of images fully consistent with the data, and so regularization is required. Regularization in the form of entropy, sparsity, or custom forms, as presently used in the field (cite EHT paper for summary). Demonstrated that higher resolution images can be obtained than CLEAN (cite Zawavzkdi). It is expected users will write programs, define objective functions, which will then be optimized using PyTorch optimization, with stochastic gradient descent or industry-grade algorithms like adam (CITE).

<!-- notes about other workflows (parametric and calibration) enabled by differentiable forward models -->
Also, parametric work akin to . Enables parametric inference, workflows similar to Galario (**CITE**), but with automatic differentiation enabled, these can be done much faster using Hamiltonian Monte Carlo. And, 1D work as well. More exotic forms of regularization, such as diffusion models, score based priors, etc. are in theory possible using the framework.

The library also provides convenience routines like classic dirty imager (varying robust values and taper) to determine starting point image and to check export of data.

Machine learning frameworks provide a natural language with which to build an expressive and realistic and differentiable forward-model of a radio interferometric dataset, which can then be optimized using stochastic gradient descent to perform image synthesis. We believe this workflow is especially powerful for scientific applications because it enables fine-scale, residual calibration physics to be parameterized and optimized simultaneously with image synthesis. Differentiable physical models have been employed to great success in other astrophysical settings such as exoplanet discovery [@bedell19] and cosmology [@campagne23].

# Documentation and case studies


MPoL is licensed under an MIT license. The main documentation is available at https://mpol-dev.github.io/MPoL/ and the code is developed in the open at https://github.com/MPoL-dev/MPoL.

Rather than an end-to-end, binary, the user is expected to use MPoL and PyTorch primitives to write short imaging scripts, in much the same way that users write PyTorch scripts for machine learning workflows [official PyTorch examples](https://github.com/pytorch/examples).

Examples are at **LINK** examples and built on a slower cadence. The codebase has been used in [@zawadzki23]. Huang for parametric modeling [@huang24].

![Left: the synthesized image produced by the DSHARP ALMA Large Program [@andrews18] using `CASA/tclean`. Right:  The regularized maximum likelihood image produced using `MPoL` on the same data. Both images are displayed using a `sqrt` stretch, with upper limit truncated to 70\% and 40\% of max value for CLEAN and `MPoL`, respectively, to emphasize faint features. The CLEAN algorithm permits negative intensity values, while the `MPoL` algorithm enforces image positivity by construction. Each side of the image is 3 arcseconds. Intensity units are shown in units of Jy/arcsec^2^.](fig.pdf)

# Similar tools

Recently, there has been significant work to design robust algorithms to image spatially resolved sources. A non-exhaustive list includes the `RESOLVE` family of algorithms, which impose Gaussian random field image priors, the multi-algorithm approach of the Event Horizon Telescope Collaboration [@eht19d] including regularized maximum likelihood techniques, and domain-specific non-parametric 1D approaches like `frank` [@jennings20]. Several approaches have leveraged deep-learning, such as score-based priors [@dia23], denoising diffusion probabilistic models [@wang23], and residual-to-residual deep neural networks [@dabbech24]. A commonality shared by most imaging alternatives is the way in which a model of the synthesized image is fed-forward to be evaluated against the visibility dataset, which contrasts with the inverse imaging and deconvolution approach employed by CLEAN.

EHT imagers, BASP group, MaxEnt Carcamo and various adherences to the framework in CASA. MPoL is focused as a PyTorch library, and works as glue to integrate with the rich systems of neural networks such as autodifferentiation and probabilistic programming languages.

Frankenstein is a 1D non-parametric tool and integrates with MPoL.

By contrast, MPoL aims to be a library, and therefore could be leveraged with the PyTorch ecosystem.

# Acknowledgements

We acknowledge funding from an ALMA Development Cycle 8 grant number AST-1519126.  ALMA is a partnership of ESO (representing its member states), NSF (USA) and NINS (Japan), together with NRC (Canada), MOST and ASIAA (Taiwan), and KASI (Republic of Korea), in cooperation with the Republic of Chile. The Joint ALMA Observatory is operated by ESO, AUI/NRAO and NAOJ. The National Radio Astronomy Observatory is a facility of the National Science Foundation operated under cooperative agreement by Associated Universities, Inc.

# References

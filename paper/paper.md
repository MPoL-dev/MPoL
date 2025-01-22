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

When an astrophysical source is observed by a radio interferometer, there are frequently large gaps in the spatial frequency coverage. Therefore, rather than perform a direct Fourier inversion, images must be synthesized from the visibility data using an imaging algorithm; it is common for the incomplete sampling to severely hamper image fidelity [@condon16; @thompson17]. CLEAN is the traditional image synthesis algorithm of the radio interferometry community [@hogbom74], with a modern implementation in the reduction and analysis software CASA [@mcmullin07; @casa22], the standard for current major facility operations [e.g., @hunter23]. CLEAN excels at the rapid imaging of astronomical fields comprising unresolved point sources (e.g. quasars) and marginally resolved sources, but may struggle when the source morphology is not well-matched by the CLEAN basis set (e.g., point sources, Gaussians), a common situation with ring-like protoplanetary disk sources [@disk20, §3].

High fidelity imaging algorithms for spatially resolved sources are needed to realize the full scientific potential of groundbreaking observatories like the Atacama Large Millimeter Array  (ALMA; @wootten09), the Event Horizon Telescope [@eht19a], and the Square Kilometer Array [@dewdney09] as they deliver significantly improved sensitivity and resolving power compared to previous generation instruments. In the field of planet formation alone, spatially resolved observations from ALMA have rapidly advanced our understanding of protoplanetary disk structures [@andrews20], kinematic signatures of embedded protoplanets [@pinte18], and circumplanetary disks [@benisty21; @casassus22]. Application of higher performance imaging techniques to these groundbreaking datasets [e.g., @casassus22] showed great promise in unlocking further scientific progress. Simultaneously, a flexible, open-source platform could integrate machine learning algorithms and computational imaging techniques from non-astronomy fields.

# The Million Points of Light (MPoL) library

`MPoL` is a library designed for feed-forward modeling of interferometric datasets using Python, Numpy [@harris20], and the computationally performant machine learning framework PyTorch [@paszke19], which debuted with @zawadzki23. `MPoL` implements a set of foundational interferometry components using PyTorch `nn.module`, which can be easily combined to build a forward-model of the interferometric dataset(s) at hand. We strive to seamlessly integrate with the PyTorch ecosystem so that users can easily leverage well-established machine learning workflows: optimization with stochastic gradient descent [@bishop23; Ch. 7], straightforward acceleration with GPU(s), and integration with common neural network architectures.

In a typical feed-forward workflow, `MPoL` users will use foundational components like `BaseCube` and `ImageCube` to define the true-sky model, Fourier layers like `FourierCube` or `NuFFT` [wrapping `torchkbnufft`; @nufft20] to apply the Fourier transform and sample the visibility function at the location of the array baselines, and the negative log likelihood to calculate a data loss. Backpropagation [see @baydin18 for a review] and stochastic gradient descent [e.g., AdamW; @loshchilov17] are used to find the true-sky model that minimizes the loss function. However, because of the aforementioned gaps in spatial frequency coverage, there is technically an infinite number of true-sky images fully consistent with the data likelihood, so regularization loss terms are required. `MPoL` supports Regularized Maximum Likelihood (RML) imaging with common regularizers like maximum entropy, sparsity, and others [e.g., as used in @eht19d]; users can also implement custom regularizers with PyTorch.

`MPoL` also provides several other workflows relevant to astrophysical research. First, by seamlessly coupling with the probabilistic programming language Pyro [@pyro19], `MPoL` supports Bayesian parametric inference of astronomical sources by modeling the data visibilities. Second, users can implement additional data calibration components as their data requires, enabling fine-scale, residual calibration physics to be parameterized and optimized simultaneously with image synthesis [following the radio interferometric measurement equation @hamaker96; @smirnov11a]. Finally, the library also provides convenience utilities like `DirtyImager` (including Briggs robust and UV taper) to confirm the data has been loaded correctly. The MPoL-dev organization also develops the [MPoL-dev/visread](https://mpol-dev.github.io/visread/) package, which is designed to facilitate the extraction of visibility data from CASA's Measurement Set format for use in alternative imaging workflows.

# Documentation, examples, and scientific results

MPoL is freely available, open-source software licensed via the MIT license and is developed on GitHub at [MPoL-dev/MPoL](https://github.com/MPoL-dev/MPoL). Installation and API documentation is hosted at [https://mpol-dev.github.io/MPoL/](https://mpol-dev.github.io/MPoL/), and is continuously built with each commit to the `main` branch. As a library, `MPoL` expects researchers to write short scripts using use `MPoL` and PyTorch primitives, in much the same way that PyTorch users write scripts for machine learning workflows (e.g., as in the [official PyTorch examples](https://github.com/pytorch/examples)). `MPoL` example projects are hosted on GitHub at [MPoL-dev/examples](https://github.com/MPoL-dev/examples). These include an introduction to generating mock data, a quickstart using stochastic gradient descent, and a Pyro workflow using stochastic variational inference (SVI) to replicate the parametric inference done in @guzman18, among others. In Figure \ref{imlup}, we compare an image obtained with CLEAN to that using `MPoL` and RML, synthesized from the data presented in @huang18b, highlighting the improvement in resolution offered by feed-forward modeling technologies.

`MPoL` has already been used in a number of scientific publications. @zawadzki23 introduced `MPoL` and explored RML imaging for ALMA observations of protoplanetary disks, finding a 3x improvement in spatial resolution at comparable sensitivity. @dia23 used `MPoL` as a reference imaging implementation to evaluate the performance of their score-based prior algorithm. @huang24 used the parametric inference capabilities of `MPoL` to analyze radial dust substructures in a suite of eight protoplanetary disks in the $\sigma$ Orionis stellar cluster. `MPoL` was selected as an imaging technology of the exoALMA large program, where Zawadzki et al. 2024 *submitted* used RML imaging to obtain high resolution image cubes of molecular line emission in protoplanetary disks in order to identify non-Keplerian features that may trace planet-disk interactions.

![Left: the synthesized image produced by the DSHARP ALMA Large Program [@andrews18] using `CASA/tclean`. Right:  The regularized maximum likelihood image produced using `MPoL` on the same data. Both images are displayed using a `sqrt` stretch, with upper limit truncated to 70\% and 40\% of max value for CLEAN and `MPoL`, respectively, to emphasize faint features. The CLEAN algorithm permits negative intensity values, while the `MPoL` algorithm enforces image positivity by construction. Each side of the image is 3 arcseconds. Intensity units are shown in units of Jy/arcsec^2^. \label{imlup}](fig.pdf)

# Similar tools

Recently, there has been significant work to design robust algorithms to image spatially resolved sources. A non-exhaustive list includes the `RESOLVE` family of algorithms [@junklewitz16], which impose Gaussian random field image priors, the multi-algorithm approach of the Event Horizon Telescope Collaboration [@eht19d] including regularized maximum likelihood techniques, MaxEnt [@carcamo18], and domain-specific non-parametric 1D approaches like `frank` [@jennings20]. Several approaches have leveraged deep-learning, such as score-based priors [@dia23], denoising diffusion probabilistic models [@wang23], and residual-to-residual deep neural networks [@dabbech24]. By contrast to many imaging software programs, `MPoL` is designed as a library, and so in theory can support a variety of forward-modeling workflows.

The parametric modeling capabilities of `MPoL`, provided by integration with `Pyro`, are similar to the `emcee` [@foreman-mackey13] + synthetic visibility workflow provided by the Galario software [@tazzari18]. Since PyTorch enables automatic differentiation, `Pyro` users can utilize HMC/NUTS sampling [@neal12; @hoffman14] or SVI, which offer significant benefits in high dimensional spaces compared to ensemble MCMC samplers.


# Acknowledgements

We acknowledge funding from an ALMA Development Cycle 8 grant number AST-1519126. J.H. acknowledges support by the National Science Foundation under Grant No. 2307916. ALMA is a partnership of ESO (representing its member states), NSF (USA) and NINS (Japan), together with NRC (Canada), MOST and ASIAA (Taiwan), and KASI (Republic of Korea), in cooperation with the Republic of Chile. The Joint ALMA Observatory is operated by ESO, AUI/NRAO and NAOJ. The National Radio Astronomy Observatory is a facility of the National Science Foundation operated under cooperative agreement by Associated Universities, Inc.

# References

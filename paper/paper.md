---
title: 'Million Points of Light (MPoL): A Python package for imaging and inference with astronomical interferometric data'
tags:
  - Python
  - astronomy
  - imaging
  - fourier
  - radio astronomy
  - radio interferometry
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
date: 12 December 2024
bibliography: paper.bib
aas-journal: Astrophysical Journal
---

* 1000 words
* A summary describing the high-level functionality and purpose of the software for a diverse, non-specialist audience.
* A clear statement of need that illustrates the purpose of the software.
* A description of how this software compares to other commonly-used packages in this research area.
* Mentions (if applicable) of any ongoing research projects using the software or recent scholarly publications enabled by it.
* A list of key references including a link to the software archive.

# Summary

Interferometric imaging is the process of recovering a spatial domain image from a Fourier domain signal that is only partially sampled. The technique is applied in a large number of fields from medical imaging to remote sensing, optics and astronomy. Within astronomy, interferometry conducted at radio, infrared and optical frequencies yields unparalleled spatial resolution in an image, corresponding to physical scales that are otherwise inaccessibly small. *Million Points of Light* (`MPoL`) is a Python package for astronomical interferometric imaging. It couples a statistical modeling framework with an efficient computational implementation to reconstruct images of astronomical sources from data measured by large telescopes such as the Atacama Large Millimeter/Submillimeter Array (ALMA). 

# Statement of need

Accurately reconstructing an image from sparse Fourier data is an ill-posed problem that remains an outstanding challenge in astronomical research, particularly in sub-mm astronomy. There, the current standard approach to interferometric imaging is `CLEAN` [@hogbom_1974; @clark_1980], an empirical, algorithmic procedure that requires a high degree of user intervention. The algorithm is not computationally efficient and thus not practical for large datasets (~100 GB) that are becoming increasingly common in the field. And the enclosing software lacks the accessibility and up-to-date documentation to easily modify the algorithm for custom use cases [@mcmullin_2007; @casa_2022]. Collectively these limitations necessitate an alternative imaging formalism and software implementation. 

# The Million Points of Light library

Built on PyTorch for idiomatic expression of radio interferometry problems, including using stochastic gradient descent.

`MPoL` is a statistically robust, nonparametric modeling approach to interferometric imaging in a user-friendly, well-documented package that is computationally performant. The software is designed to be applied to reconstruction of an individual image or an entire 'cube' of tens to hundreds of images of an astronomical source observed at different frequencies. The images obtained are of simultaneously higher spatial resolution and sensitivity than their counterparts produced by `CLEAN`. Programatically, `MPoL` is built on `PyTorch`, using its auto-differentiation capabilities to drive likelihood optimization with gradient descent and its parallelization support to optionally accelerate the imaging workflow on GPUs and TPUs. The imaging framework in `MPoL` is also flexible, with the ability to easily add alternative or additional priors into likelihood calculation. Extensions to the core functionality are actively developed, such as the recent implementation of parametric inference with `Pyro`, as are further optimizations to the core routines.

# Documentation and case studies

The main documentation is available at https://mpol-dev.github.io/MPoL/

The codebase has been used in [@zawadzki_2023].

![Left: the synthesized image produced by the DSHARP ALMA Large Program [@andrews18] using \texttt{CASA/tclean}. Right:  The regularized maximum likelihood image produced using \texttt{MPoL} on the same data. Both images are displayed using a `sqrt` stretch, with upper limit truncated to 70\% and 40\% of max value for CLEAN and \texttt{MPoL}, respectively, to emphasize faint features. The CLEAN algorithm permits negative intensity values, while the `MPoL` algorithm enforces image positivity by construction. Image dimensions are 3 arcseconds to a side. Intensity units are shown in units of Jy/arcsec^2^.](fig.pdf)

# Similar tools

EHT imagers, BASP group, MaxEnt Carcamo and various adherences to the framework in CASA. MPoL is focused as a PyTorch library, and works as glue to integrate with the rich systems of neural networks such as autodifferentiation and probabilistic programming languages.

Frankenstein is a 1D non-parametric tool and integrates with MPoL.

# Acknowledgements

We acknowledge funding from an ALMA Development Cycle 8 grant number AST-1519126.  ALMA is a partnership of ESO (representing its member states), NSF (USA) and NINS (Japan), together with NRC (Canada), MOST and ASIAA (Taiwan), and KASI (Republic of Korea), in cooperation with the Republic of Chile. The Joint ALMA Observatory is operated by ESO, AUI/NRAO and NAOJ. The National Radio Astronomy Observatory is a facility of the National Science Foundation operated under cooperative agreement by Associated Universities, Inc.

# References

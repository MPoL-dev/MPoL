---
title: 'MPoL: A Python package for scalable, nonparametric interferometric imaging'
tags:
  - Python
  - astronomy
  - imaging
  - fourier
authors:
  - name: Ian Czekala
    orcid: 0000-0002-1483-8811
    # equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    # corresponding: true
    affiliation: 1
    # affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Jeff Jennings
    orcid: 0000-0002-7032-2350
    affiliation: 2
  - name: Brianna Zawadzki
    orcid: 0000-0001-9319-1296
    affiliation: 3
  - name: Ryan Loomis
    orcid: 0000-0002-8932-1219
    affiliation: 4
  - name: Kadri Nizam
    orcid: 0000-0002-7217-446X
    affiliation: 2
  - name: Megan Delamer
    # orcid: # can't find
    affiliation: 2     
  - name: Kaylee de Soto
    orcid: 0000-0002-9886-2834
    affiliation: 2
  - name: Robert Frazier
    orcid: 0000-0001-6569-3731
    affiliation: 2      
  - name: Hannah Grzybowski
    # orcid: # can't find
    affiliation: 2         
  - name: Mary Ogborn
    orcid: 0000-0001-9741-2703
    affiliation: 2
  - name: Tyler Quinn
    orcid: 0000-0002-8974-8095
    affiliation: 2
affiliations:
 - name: University of St Andrews, Scotland # "institution, country" format suggested by JOSS
   index: 1
 - name: Pennsylvania State University, USA
   index: 2
 - name: Wesleyan University, USA
   index: 3
 - name: National Radio Astronomy Observatory, USA
   index: 4      
date: 14 November 2023
bibliography: paper.bib
---

# Summary

Interferometric imaging is the process of recovering a spatial domain image from a Fourier domain signal that is only partially sampled. The technique is applied in a large number of fields from medical imaging to remote sensing, optics and astronomy. Within astronomy, interferometry conducted at radio, infrared and optical frequencies yields unparalleled spatial resolution in an image, corresponding to physical scales that are otherwise inaccessibly small. `Million Points of Light` (`MPoL`) is a Python package for astronomical interferometric imaging. It couples a statistical modeling framework with an efficient computational implementation to reconstruct images of astronomical sources from data measured by large telescopes such as the Atacama Large Millimeter/Submillimeter Array (ALMA). 

# Statement of need

Accurately reconstructing an image from sparse Fourier data is an ill-posed problem that remains an outstanding challenge in astronomical research, particularly in sub-mm astronomy. There, the current standard approach to interferometric imaging is `CLEAN` [@Hogbom_1974; @Clark_1980], an empirical, algorithmic procedure that requires a high degree of user intervention. The algorithm is not computationally efficient and thus not practical for large datasets (~100 GB) that are becoming increasingly common in the field. And the enclosing software lacks the accessibility and up-to-date documentation to easily modify the algorithm for custom use cases [@McMullin_2007; @CASA_2022]. Collectively these limitations necessitate an alternative imaging formalism and software implementation. 

`MPoL` is a statistically robust, nonparametric modeling approach to interferometric imaging in a user-friendly, well-documented package that is computationally performant. The software is designed to be applied to reconstruction of an individual image or an entire 'cube' of tens to hundreds of images of an astronomical source observed at different frequencies. The images obtained are of simultaneously higher spatial resolution and sensitivity than their counterparts produced by `CLEAN`. Programatically, `MPoL` is built on `PyTorch`, using its auto-differentiation capabilities to drive likelihood optimization with gradient descent and its parallelization support to optionally accelerate the imaging workflow on GPUs and TPUs. The imaging framework in `MPoL` is also flexible, with the ability to easily add alternative or additional priors into likelihood calculation. Extensions to the core functionality are actively developed, such as the recent implementation of parametric inference with `Pyro`, as are further optimizations to the core routines.

`MPoL` is used in astrophysical research to image and study objects such as protoplanetary disks and Solar System bodies. It is currently being applied to multiple projects, from individual use cases to large collaborations. The software could be applied without modification to research in other subfields of astronomy that use data from sub-mm interferometers, including cosmology, extragalactic astronomy, and star formation. With a reasonable amount of modification, it could be adopted for datasets obtained by infrared and optical interferometers, or to interferometric imaging problems beyond astronomy.

# Acknowledgements

<!-- We acknowledge contributions from -->

# References
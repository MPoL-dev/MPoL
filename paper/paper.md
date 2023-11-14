---
title: 'MPoL: A Python package for interferometric imaging'
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
  - name:Jeff Jennings
    orcid: 0000-0002-7032-2350
    corresponding: true
    affiliation: 2
  # - name:Brianna Zawadzki
  #   orcid: 0000-0001-9319-1296
  #   affiliation: 3
  # - name:Ryan Loomis
  #   orcid: 0000-0002-8932-1219
  #   affiliation: 4
affiliations:
 - name: University of St Andrews, Scotland
   index: 1
 - name: Pennsylvania State University, USA
   index: 2
#  - name: Wesleyan University, USA
#    index: 3
#  - name: National Radio Astronomy Observatory, USA
#    index: 4      
date: 14 November 2023
bibliography: paper.bib
---

# Summary

Interferometric imaging is the process of recovering a spatial domain image from a Fourier domain signal that is only partially sampled. The technique is applied in a large number of fields from medical imaging to remote sensing, optics and astronomy. Within astronomy, interferometry conducted at radio, infrared and optical frequencies yields unparalleled spatial resolution in an image, corresponding to physical scales that are otherwise inaccessibly small. `Million Points of Light` (`MPoL`) is a Python package for astronomical interferometric imaging. It couples a statistical modeling framework with an efficient computational implementation to reconstruct images of astronomical sources from data measured by large telescopes such as the Atacama Large Millimeter/Submillimeter Array (ALMA). 

# Statement of need

`Million Points of Light` (`MPoL`)

# Mathematics

<!-- Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text. -->

# Citations

<!-- Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)" -->

# Figures
<!-- 
Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% } -->

# Acknowledgements

<!-- We acknowledge contributions from xx. -->

# References
---
jupytext:
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
%matplotlib inline
%run notebook_setup
```

# Using the NUFFT to get loose visibilities

The basic operation of MPoL is to  normally works by gridding.

This tutorial explains how to use the non-uniform FFT to take an image plane model to the loose visibilities of the dataset.

Different modes (parallelizing over channel with coil or with batch dimensionality).


# timing tests -- possible with actual dataset

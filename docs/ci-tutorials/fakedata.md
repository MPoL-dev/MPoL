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
%run notebook_setup
```

# Making a Mock Dataset

This is most useful if you already have a real dataset, with real baseline distributions and noise weights. Alternatively, you could acquire some baseline distribution and noise distribution, possibly using CASA's simobserve.



For this, you would take a realistic, known, sky-brightness distribution and then propagate this to the visibilities. For example, you could use an image from a simulation, a parametric model, or even an image from the Internet.

```{code-cell}
url="https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/The_final_ALMA_antenna.jpg/2560px-The_final_ALMA_antenna.jpg"
```

How many pixels does it have?

The routine just takes an Image cube, u,v, weights and produces visibilites with noise.

But there's another concern about how to put the image cube in, right? I guess that's just a matter of matching the image cube to the size. You may want to pad the image, though. You probably also want to convolve it.

Here is where it would be helpful to have a note about how changing pixel size and image dimensions affects the uv coverage. There needs to be some match up between the image and the uv size.

We'll use the same u,v distribution and noise distribution from the mock dataset. The max baseline

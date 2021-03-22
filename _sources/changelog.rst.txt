   .. _changelog-reference-label:

Changelog
=========

v0.1.0
------

* Updated citations to include Brianna Zawadzki
* Added :class:`~mpol.gridding.Gridder` and :class:`~mpol.gridding.GridCoords` objects.
* Removed ``mpol.dirty_image`` module.
* Migrated prolate spheroidal wavefunctions to ``mpol.spheroidal_gridding`` module.
* Added Jupyter notebook tutorial build process using Jupytext
* Added SimpleNet precomposed module
* Added Mermaid.js charting ability (for flowcharts)
* Moved docs to github.io pages instead of Read the docs
* Added Jy/arcsec^2 units to Gridder output.

v0.0.5
------

* Introduced this Changelog
* Updated citations to include Ryan Loomis
* Added `dirty_image.get_dirty_image` routine, which includes Briggs robust weighting.
* Added assert statements to catch if the user chooses `cell_size` too coarsely relative to the spatial frequencies in the dataset.
* Implemented preliminary power spectral density loss functions.
* The image cube is now natively stored (and optimized) using the natural logarithm of the pixel values. This defacto enforces positivity on all pixel values.
* Changed entropy function to follow EHT-IV.

v0.0.4
------

* Made the package ``pip`` installable.
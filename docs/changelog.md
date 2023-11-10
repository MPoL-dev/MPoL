(changelog-reference-label)=

# Changelog

## v0.2.0

- Moved docs build out of combined and into standalone test workflow
- Updated package test workflow with new dependencies and caching
- Added geometry tests
- Reorganized some of the docs API
- Expanded discussion and demonstration in `optimzation.md` tutorial
- Localized harcoded Zenodo record reference to single instance, and created new external Zenodo record from which to draw
- Added [Parametric inference with Pyro tutorial](large-tutorials/pyro.md) 
- Updated some discussion and notation in `rml_intro.md` tutorial
- Added `mypy` static type checks
- Added `frank` as a 'test' and 'analysis' extras dependency
- Added `fast-histogram` as a core dependency
- Updated support to recent Python versions
- Removed `mpol.coordinates._setup_coords` helper function from {class}`~mpol.coordinates.GridCoords`
- Added new program `mpol.crossval` with the new {class}`~mpol.crossval.CrossValidate` for running a cross-validation loop and the new {class}`~mpol.crossval.RandomCellSplitGridded` for splitting data into training and test sets
- Moved and rescoped {class}`~mpol.datasets.KFoldCrossValidatorGridded` to {class}`~mpol.crossval.DartboardSplitGridded` with some syntax changes
- Altered {class}`~mpol.datasets.GriddedDataset` to subclass from `torch.nn.Module`, altered its args, added PyTorch buffers to it, added {func}`mpol.datasets.GriddedDataset.forward` to it
- Added class method `from_image_properties` to various classes including {class}`~mpol.images.BaseCube` and {class}`~mpol.images.ImageCube`
- Altered {class}`~mpol.datasets.UVDataset` to subclass from `torch.utils.data.Dataset`, altered its initialization signature, added new properties 
- Altered {class}`~mpol.fourier.FourierCube` args and initialization signature, added PyTorch buffers to it
- Added {func}`~mpol.fourier.get_vis_residuals`
- Added new program `mpol.geometry` with new {func}`~mpol.geometry.flat_to_observer` and {func}`~mpol.geometry.observer_to_flat`
- Replaced {class}`~mpol.gridding.Gridder` with the rescoped {class}`~mpol.gridding.GridderBase` and two classes which subclass this, {class}`~mpol.gridding.DirtyImager` and {class}`~mpol.gridding.DataAverager`
- Added property `flux` to {class}`~mpol.images.ImageCube`
- Added new program `mpol.onedim` with new {func}`~mpol.onedim.radialI` and {func}`~mpol.onedim.radialV`
- Added new program `mpol.training` with new {class}`~mpol.training.TrainTest` and {func}`~mpol.onedim.radialV`
- Added new utility functions {func}`~mpol.utils.torch2npy`, {func}`~mpol.utils.check_baselines`, {func}`~mpol.utils.get_optimal_image_properties`
- Added expected types and error checks in several places throughout codebase, as well as new programs - `mpol.exceptions` and `mpol.protocols`
- Updated tests in several places and added many new tests
- Added shell script `GPU_SLURM.sh` for future test implementations
- Updated citations to include new contributors

## v0.1.4

- Removed the `GriddedResidualConnector` class and the `src/connectors.py` module. Moved `index_vis` to `datasets.py`.
- Changed BaseCube, ImageCube, and FourierCube initialization signatures

## v0.1.3

- Added the {func}`mpol.fourier.make_fake_data` routine and the [Mock Data tutorial](ci-tutorials/fakedata.md).
- Fixed a bug in the [Dirty Image Initialization](ci-tutorials/initializedirtyimage.md) tutorial so that the dirty image is delivered in units of Jy/arcsec^2.

## v0.1.2

- Switched documentation backend to [MyST-NB](https://myst-nb.readthedocs.io/en/latest/index.html).
- Switched documentation theme to [Sphinx Book Theme](https://sphinx-book-theme.readthedocs.io/en/latest/index.html).
- Added {class}`~mpol.fourier.NuFFT` layer, allowing the direct forward modeling of un-gridded $u,v$ data. Closes GitHub issue [#17](https://github.com/MPoL-dev/MPoL/issues/17).

## v0.1.1

- Added {class}`~mpol.images.HannConvCube`, incorporating Hann-like pixels and bundled it in the {class}`~mpol.precomposed.SimpleNet` module
- Added {class}`~mpol.datasets.Dartboard` and {class}`~mpol.datasets.KFoldCrossValidatorGridded` for cross validation
- Added cross validation tutorial
- Removed DatasetConnector in favor of {func}`~mpol.losses.nll_gridded`
- Added {func}`~mpol.utils.ground_cube_to_packed_cube`, {func}`~mpol.utils.packed_cube_to_ground_cube`, {func}`~mpol.utils.sky_cube_to_packed_cube`, and {func}`~mpol.utils.packed_cube_to_sky_cube`

## v0.1.0

- Updated citations to include Brianna Zawadzki
- Added {class}`~mpol.gridding.Gridder` and {class}`~mpol.gridding.GridCoords` objects
- Removed `mpol.dirty_image` module
- Migrated prolate spheroidal wavefunctions to `mpol.spheroidal_gridding` module
- Added Jupyter notebook tutorial build process using Jupytext
- Added {class}`~mpol.precomposed.SimpleNet` precomposed module
- Added Mermaid.js charting ability (for flowcharts)
- Moved docs to github.io pages instead of Read the docs
- Added $\mathrm{Jy\;arcsec}^{-2}$ units to Gridder output

## v0.0.5

- Introduced this Changelog
- Updated citations to include Ryan Loomis
- Added `dirty_image.get_dirty_image` routine, which includes Briggs robust weighting.
- Added assert statements to catch if the user chooses `cell_size` too coarsely relative to the spatial frequencies in the dataset.
- Implemented preliminary power spectral density loss functions.
- The image cube is now natively stored (and optimized) using the natural logarithm of the pixel values. This defacto enforces positivity on all pixel values.
- Changed entropy function to follow EHT-IV.

## v0.0.4

- Made the package `pip` installable.

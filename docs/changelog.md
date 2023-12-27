(changelog-reference-label)=

# Changelog

## v0.2.1

- *Placeholder* Planned changes described by Architecture GitHub Project.
- Removed convenience classmethods `from_image_properties` from across the code base. From [#233](https://github.com/MPoL-dev/MPoL/issues/233). The recommended workflow is to create a :class:`mpol.coordinates.GridCoords` object and pass that to instantiate these objects as needed. For nearly all but trivially short workflows, this simplifies the number of variables the user needs to keep track and pass around revealing the central role of the :class:`mpol.coordinates.GridCoords` object and its useful attributes for image extent, visibility extent, etc. Most importantly, this significantly reduces the size of the codebase and the burden to maintain and document multiple entry points to key `nn.modules`. We removed `from_image_properties` from
  - :class:`mpol.datasets.GriddedDataset`
  - :class:`mpol.datasets.Dartboard` 
  - :class:`mpol.fourier.NuFFT`
  - :class:`mpol.fourier.NuFFTCached` 
  - :class:`mpol.fourier.FourierCube` 
  - :class:`mpol.gridding.GridderBase` 
  - :class:`mpol.gridding.DataAverager`
  - :class:`mpol.gridding.DirtyImager`
  - :class:`mpol.images.BaseCube`
  - :class:`mpol.images.ImageCube`
- Removed unused routine `mpol.utils.log_stretch`.
- Added type hints for core modules ([#54](https://github.com/MPoL-dev/MPoL/issues/54)). This should improve stability of core routines and help users when writing code using MPoL in an IDE.
- Manually line wrapped many docstrings to conform to 88 characters per line or less. Ian thought `black` would do this by default, but actually that [doesn't seem to be the case](https://github.com/psf/black/issues/2865).
- Fully leaned into the `pyproject.toml` setup to modernize build via [hatch](https://github.com/pypa/hatch). This centralizes the project dependencies and derives package versioning directly from git tags. Intermediate packages built from commits after the latest tag (e.g., `0.2.0`) will have an extra long string, e.g., `0.2.1.dev178+g16cfc3e.d20231223` where the version is a guess at the next version and the hash gives reference to the commit. This means that developers bump versions entirely by tagging a new version with git (or more likely by drafting a new release on the [GitHub release page](https://github.com/MPoL-dev/MPoL/releases)).
- Removed `setup.py`.
- TOML does not support adding keyed entries, so creating layered build environments of default, `docs`, `test`, and `dev` as we used to with `setup.py` is laborious and repetitive with `pyproject.toml`. We have simplified the list to be default (key dependencies), `test` (minimal necessary for test-suite), and `dev` (covering everything needed to build the docs and actively develop the package).
- Removed custom `spheroidal_gridding` routines, tests, and the `UVDataset` object that used them. These have been superseded by the TorchKbNuFFT package. For reference, the old routines (including the tricky `corrfun` math) is preserved in a Gist [here](https://gist.github.com/iancze/f3d2769005a9e2c6731ee6977f166a83).
- Changed API of {class}`~mpol.fourier.NuFFT`. Previous signature took `uu` and `vv` points at initialization (`__init__`), and the `.forward` method took only an image cube. This behaviour is preserved in a new class {class}`~mpol.fourier.NuFFTCached`. The updated signature of {class}`~mpol.fourier.NuFFT` *does not* take `uu` and `vv` at initialization. Rather, its `forward` method is modified to take an image cube and the `uu` and `vv` points. This allows an instance of this class to be used with new `uu` and `vv` points in each forward call. This follows the standard expectation of a layer (e.g., a linear regression function predicting at new `x`) and the pattern of the TorchKbNuFFT package itself. It is expected that the new `NuFFT` will be the default routine and `NuFFTCached` will only be used in specialized circumstances (and possibly deprecated/removed in future updates). Changes implemented by [#232](https://github.com/MPoL-dev/MPoL/pull/232).
- Moved "Releasing a new version of MPoL" from the wiki to the Developer Documentation ({ref}`releasing-new-version-label`).

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

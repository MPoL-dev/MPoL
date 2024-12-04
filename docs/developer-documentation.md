(developer-documentation-label)=

# Developer Documentation

If you find an issue with the code, documentation, or would like to make a specific suggestion for an improvement, please raise an issue on the [Github repository](https://github.com/MPoL-dev/MPoL/issues). If you have a more general query or would just like to discuss a topic, please make a post on our [Github discussions page](https://github.com/MPoL-dev/MPoL/discussions).

If you are new to contributing to an open source project, we recommend a quick read through the excellent contributing guides of the [exoplanet](https://docs.exoplanet.codes/en/stable/user/dev/) and [astropy](https://docs.astropy.org/en/stable/development/workflow/development_workflow.html) packages. What follows in this guide draws upon many of the suggestions from those two resources. There are many ways to contribute to an open source project like MPoL, and all of them are valuable to us. No contribution is too small---even a typo fix is appreciated!

The MPoL source repository is hosted on [Github](https://github.com/MPoL-dev/MPoL) as part of the [MPoL-dev](https://github.com/MPoL-dev/) organization. We use a "fork and pull request" model for collaborative development. If you are unfamiliar with this workflow, check out this short Github guide on [forking projects](https://guides.github.com/activities/forking/). 

## Development dependencies

Extra packages required for development can be installed via

```
(venv) $ pip install -e ".[dev]"
```

This directs pip to install whatever package is in the current working directory (`.`) as an editable package (`-e`), using the set of `[dev]` optional packages. There is also a more limited set of packages under `[test]`. You can view these packages in the `pyproject.toml` file. 

## Testing

MPoL includes a test suite written using [pytest](https://docs.pytest.org/). We aim for this test suite to be as comprehensive as possible, since this helps us achieve our goal of shipping stable software.

### Running tests

To run all of the tests, change to the root of the repository and invoke

```
$ python -m pytest
```

If a test errors (especially on the `main` branch), please report what went wrong as a bug report issue on the [Github repository](https://github.com/MPoL-dev/MPoL/issues).

### Viewing test and debug plots

Some tests produce temporary files, like plots, that could be useful to view for development or debugging. Normally these are saved to a temporary directory created by the system which will be cleaned up after the tests finish. To preserve them, first create a plot directory (e.g., `plotsdir`) and then run the tests with this `--basetemp` specified

```
$ mkdir plotsdir
$ python -m pytest --basetemp=plotsdir
```

### Test coverage

To investigate how well the test suite covers the full range of program functionality, you can run [coverage.py](https://coverage.readthedocs.io/en/coverage-5.5/) through pytest using [pytest-cov](https://pypi.org/project/pytest-cov/), which should already be installed as part of the `[test]` dependencies

```
$ pytest --cov=mpol
$ coverage html
```

And then use your favorite web browser to open `htmlcov/index.html` and view the coverage report.

For more information on code coverage, see the [coverage.py documentation](https://coverage.readthedocs.io/en/coverage-5.5/). A worthy goal is to reach 100% code coverage with the testing suite. However, 100% coverage *doesn't mean the code is free of bugs*. More important than complete coverage is writing tests that properly probe program functionality.

### Test cache

Several of the tests require mock data that is not practical to package within the github repository itself, and so it is stored on Zenodo and downloaded using astropy caching functions. If you run into trouble with the test cache becoming stale, you can delete it by deleting the `.mpol/cache` folder in your home directory.

### Contributing tests
MPoL tests are located within the `test/` directory and follow [pytest](https://docs.pytest.org/en/6.2.x/contents.html#toc) conventions. Please add your new tests to this directory---we love new and useful tests.

If you are adding new code functionality to the package, please make sure you have also written a set of corresponding tests probing the key interfaces. If you submit a pull request implementing code functionality without any new tests, be prepared for 'tests' to be one of the first suggestions on your pull request. Some helpful advice on *which* tests to write is [here](https://docs.python-guide.org/writing/tests/), [here](https://realpython.com/pytest-python-testing/), and [here](https://www.nerdwallet.com/blog/engineering/5-pytest-best-practices/).


## Type hinting

Core MPoL routines are type-checked with [mypy](https://mypy.readthedocs.io/en/stable/index.html) for 100% coverage. Before you push your changes to the repo, you will want to make sure your code passes type checking locally (otherwise they will fail the GitHub Actions continuous integration tests). You can do this from the root of the repo by 

```
mypy src/mpol --pretty
```

If you are unfamiliar with typing in Python, we recommend reading the [mypy cheatsheet](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html) to get started.


(documentation-build-reference-label)=
## Documentation

MPoL documentation is written as docstrings attached to MPoL classes and functions (using reSt) and as individual `.md` files located in the `docs/` folder. The documentation is built using the [Sphinx](https://www.sphinx-doc.org/en/master/) Python documentation generator, with the [MyST-NB](https://myst-nb.readthedocs.io/en/latest/index.html) plugin.


### Building documentation

To build the documentation, change to the `docs/` folder and run

```
$ make html
```

If successful, the HTML documentation will be available in the `_build/html` folder. You can preview the documentation locally using your favorite web browser by opening up `_build/html/index.html`

You can clean up (delete) all of the built products by running

```
$ make clean
```

### Contributing tutorials

If your tutorial is self-contained and has minimal computational needs (under 30 seconds on a single CPU), provide the source file as a [MyST-NB `.md` file](https://myst-nb.readthedocs.io/en/latest/authoring/basics.html#text-based-notebooks) so that it is built with MyST-NB on GitHub Actions. You can use the small tutorials in `docs/ci-tutorials` as a reference to get started.

If your tutorial requires significant computational resources (e.g., a GPU, multiple CPS, or more than 30 seconds runtime), execute the notebook on your local computing resources and commit both the `.md` and `.ipynb` files (with output cells) directly to the repository. You can see examples for these in `docs/large-tutorials`.

If you're thinking about contributing a tutorial and would like guidance on form or scope, please raise an [issue](https://github.com/MPoL-dev/MPoL/issues) or [discussion](https://github.com/MPoL-dev/MPoL/discussions) on the github repository.

### Older documentation versions

In the rare situation where you require documentation for a different (older) version of MPoL, you can swap to an older tag

```
git fetch --tags
git checkout tags/v0.2.0
```

and then build the documentation.

## Releasing a new version of MPoL

It is our intent that the `main` branch of the github repository always reflects a stable version of the code that passes all tests. After significant new functionality has been introduced, a tagged release (e.g., `v0.1.1`) is generated from the main branch and pushed to PyPI.

To do this, follow this checklist in order:

1. Ensure *all* tests are passing on your PR, both locally and on GitHub Actions.
2. Ensure the docs build locally without errors or warnings. Check output by opening `docs/_build/html/index.html` with your web browser.
3. Perform final edits documenting the changes since last version in `docs/changelog.md`. Highlight potentially breaking changes and suggest how users might update their workflow.
4. Check contributors in `CONTRIBUTORS.md` up to date.
5. Update the copyright year and citation in `README.md`
    * In the citation, update all fields except 'Zenodo', 'doi', and 'url' (the current DOI will cite all versions and the URL will direct to the most recent version)
6. Merge your PR into `main` using the GitHub interface.
    * A new round of tests will be triggered by the merge. Make sure *all* of these pass.
7. Go to the [Releases](https://github.com/MPoL-dev/MPoL/releases) page, draft release notes, and publish a pre-release
    * Ensure the `pre-release.yml` workflow passes.
8. Publish the true release. GitHub actions will automatically uploaded the built package to PyPI and archive it on Zenodo
    * Verify the `package.yml` workflow passed.

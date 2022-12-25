(developer-documentation-label)=

# Developer Documentation

If you find an issue with the code, documentation, or would like to make a specific suggestion for an improvement, please raise an issue on the [Github repository](https://github.com/MPoL-dev/MPoL/issues). If you have a more general query or would just like to discuss a topic, please make a post on our [Github discussions page](https://github.com/MPoL-dev/MPoL/discussions).

If you are new to contributing to an open source project, we recommend a quick read through the excellent contributing guides of the [exoplanet](https://docs.exoplanet.codes/en/stable/user/dev/) and [astropy](https://docs.astropy.org/en/stable/development/workflow/development_workflow.html) packages. What follows in this guide draws upon many of the suggestions from those two resources. There are many ways to contribute to an open source project like MPoL, and all of them are valuable to us. No contribution is too small---even a typo fix is appreciated!

## Release model

The MPoL project values stable software, and so we place a special emphasis on writing and running tests to ensure the code works reliably for as many users as possible. It is our intent that the `main` branch of the github repository always reflects a stable version of the code that passes all tests. After significant new functionality has been introduced, a tagged release (e.g., `v0.1.1`) is generated from the main branch and pushed to PyPI.

## Setting up your development environment

### Forking MPoL and cloning from Github

MPoL is developed using the git version control system. The source repository is hosted on [Github](https://github.com/MPoL-dev/MPoL) as part of the [MPoL-dev](https://github.com/MPoL-dev/) organization. If you are new to using git and Github, we recommend reviewing the git resources on the [astropy contributing guide](https://docs.astropy.org/en/stable/development/workflow/development_workflow.html).

We use a "fork and pull request" model for collaborative development. If you are unfamiliar with this workflow, check out this short Github guide on [forking projects](https://guides.github.com/activities/forking/). For even more reference material, see the official Github documentation on [collaborating with issues and pull requests](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests).

After forking the MPoL repository to your own github account, clone the repository to your local machine.

```
$ git clone https://github.com/YOURUSERNAME/MPoL.git
```

### Python virtual environment

To keep things organized, we recommend creating a Python [virtual environment](https://docs.python.org/3/tutorial/venv.html) specifically for MPoL development (though this step isn't strictly necessary). To do this,

```
$ cd MPoL
$ python3 -m venv venv
```

More information on virtual environments is available through the official [Python documentation](https://docs.python.org/3/tutorial/venv.html). Then, activate the virtual environment

```
$ source venv/bin/activate
```

This gives you a clean Python environment in which to work without fear of conflicting package dependencies (e.g., perhaps you want to have two versions of MPoL installed on your system, one for production work and the other for development work. This is possible as long as you use two different virtual environments). You will need to remember to activate the corresponding virtual environment with each new shell session, however.

### Installing MPoL and development dependencies

Then, use pip to install the MPoL package. We recommend installing the package using

```
(venv) $ pip install -e ".[dev]"
```

This directs pip to install whatever package is in the current working directory (`.`) as an editable package (`-e`). More information on the pip command line arguments is available via `pip install --help`. We've additionally told pip to install the packages listed in the (`[dev]`) `EXTRA_REQUIRES` variable in [setup.py](https://github.com/MPoL-dev/MPoL/blob/main/setup.py). This variable contains a list of all of the Python packages that you might need throughout the course of developing and testing the package (e.g., pytest). These packages are only needed when developing, building, and testing the package. They are not needed to run the package in its normal configuration, hence, they are not installed with the vanilla `pip install MPoL`.

We also recommend using a set of git [pre-commit hooks](https://pre-commit.com/) that we have configured via the `.pre-commit-config.yaml` file in the MPoL repository. They are very useful tools for keeping git diffs orderly and small. To install these tools to your local copy of the MPoL repository, run

```
$ pre-commit install
```

Then, each time you run `git commit` these scripts will run on the repository code and check for things like whether your code complies with formatting guidelines. If any issues are found, the pre-commit script will stop the commit and correct the offending files for you. Then, you can redo the `git commit` with the modified files.

(testing-reference-label)=

## Testing

MPoL includes a test suite written using [pytest](https://docs.pytest.org/). We aim for this test suite to be as comprehensive as possible, since this helps us achieve our goal of shipping stable software.

### Installing dependencies

If you are only interested in running the tests, you can install the more limited set of testing package dependencies via

```
$ pip install ".[test]"
```

after you've cloned the repository and changed to the root of the repository. Otherwise, we recommend following the development environment instructions above, since the `[dev]` list is a superset of the `[test]` list.

### Running the tests

To run all of the tests, change to the root of the repository and invoke

```
$ python -m pytest
```

If a test errors (especially on the `main` branch), please report what went wrong as a bug report issue on the [Github repository](https://github.com/MPoL-dev/MPoL/issues).

### Test cache

Several of the tests require mock data that is not practical to package within the github repository itself, and so it is stored on Zenodo and downloaded using astropy caching functions. If you run into trouble with the test cache becoming stale, you can delete it by deleting the `.mpol/cache` folder in your home directory.

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

## Documentation

MPoL documentation is written as docstrings attached to MPoL classes and functions and as individual `.rst` or `.md` files located in the `docs/` folder. The documentation is built using the [Sphinx](https://www.sphinx-doc.org/en/master/) Python documentation generator, with the [MyST-NB](https://myst-nb.readthedocs.io/en/latest/index.html) plugin.

### Dependencies

If you are only interested in building the documentation, you can install the more limited set of documentation package dependencies via

```
$ pip install ".[docs]"
```

after you've cloned the repository and changed to the root of the repository. Otherwise, we recommend following the development environment instructions above, since the `[dev]` list is a superset of the `[docs]` list.

### Building the Documentation

To build the documentation, change to the `docs/` folder and run

```
$ make html
```

If successful, the HTML documentation will be available in the `_build/html` folder. You can preview the documentation locally using your favorite web browser by opening up `_build/html/index.html`

You can clean up (delete) all of the built products by running

```
$ make clean
```

For more information on the build process, take a look at the documentation makefile in `docs/Makefile`.

## Contributing

The following subsections describe recommended workflows for contributing code, documentation, and tutorials to the MPoL package. They are written assuming that you are familiar with the preceding development guidelines in this document.

In keeping with the [fork and pull request model](https://guides.github.com/activities/forking/), you carry out development on a forked version of the repository (i.e., `https://github.com/YOURUSERNAME/MPoL.git`) and, once you're satisfied with your changes (and all code passes the tests), you initiate a pull request back to the central repository (`https://github.com/MPoL-dev/MPoL.git`).

In general, we envision the contribution lifecycle following a pattern:

1. If you notice that the MPoL-dev repository has newer changes since you made your fork, fetch upstream changes to your repository and merge them into the `main` branch. You can do this via the Github interface by clicking "fetch upstream" and then pulling the changes to your local machine with `git pull`. Alternatively, you can do this from the command line by configuring the [remote upstream repository](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/configuring-a-remote-for-a-fork) to `https://github.com/MPoL-dev/MPoL.git` and [syncing the changes](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/syncing-a-fork) from the upstream repository to your forked repo.
2. Checkout the updated `main` branch of your MPoL repository and verify that it contains any new changes by comparing commits with `git log`.
3. Run the test suite (with `pytest`) and verify that all tests pass. If not, raise a Github issue with the error.
4. Create a new feature branch with `git checkout -b my_feature_branch`.
5. Develop your code/documentation/tutorial (see below) and commit your changes to the repository as you go. Be sure to commit the right things to the repository (such as source files) and avoid committing build products (like .png files produced as part of the documentation build, for example).
6. When you are satisfied with your changes, run the tests and build the documentation. If anything fails due to changes you made, please fix the code so that the tests pass and the documentation builds, then recommit your changes. If you cannot figure out why the tests or documentation are breaking, please raise a Github issue with the error.
7. Push the changes on your feature branch to Github with `git push origin my_feature_branch`. Make sure that the automated tests and documentation build as part of Github actions complete successfully. If not, assess the error(s), make additional changes to your feature branch, and re-push the changes until all tests pass.
8. After the tests have completed successfully, use the Github interface to initiate a pull request back to the central repository. If you know that your feature branch isn't ready to be merged, but would still like feedback on your work, please submit a [draft or "work in progress"](https://github.blog/2019-02-14-introducing-draft-pull-requests/) pull request.
9. Someone will review your pull request and may suggest additional changes for improvements. If approved, your pull request will be merged into the MPoL-dev/MPoL repo. Thank you for your contribution!

### Contributing code and tests

We strive to release a useable, stable software package. One way to help accomplish this is through writing rigorous and complete tests, especially after adding new functionality to the package. MPoL tests are located within the `test/` directory and follow [pytest](https://docs.pytest.org/en/6.2.x/contents.html#toc) conventions. Please add your new tests to this directory---we love new and useful tests.

If you are adding new code functionality to the package, please make sure you have also written a set of corresponding tests probing the key interfaces. If you submit a pull request implementing code functionality without any new tests, be prepared for 'tests' to be one of the first suggestions on your pull request. Some helpful advice on *which* tests to write is [here](https://docs.python-guide.org/writing/tests/), [here](https://realpython.com/pytest-python-testing/), and [here](https://www.nerdwallet.com/blog/engineering/5-pytest-best-practices/).

### Contributing documentation

A general workflow for writing documentation might look like

1. run `make html` in the `docs/` folder
2. look at the built documentation with your web browser
3. write/edit documentation source as docstrings, `*.rst`, or `*.md` files
4. run `make html` to rebuild those files you've changed
5. go to #2 and repeat as necessary

The idea behind [GNU make](https://www.gnu.org/software/make/manual/make.html) is that every invocation of `make html` will only rebuild the files whose dependencies have changed, saving lots of time.

### Contributing tutorials

If your tutorial is self-contained in scope and has limited computational needs (will complete on a single CPU in under 30 seconds), we recommend you provide the source file as a [MyST-NB `.md` file](https://myst-nb.readthedocs.io/en/latest/authoring/basics.html#text-based-notebooks) so that we can build and test it as part of the continuous integration github workflow. If your tutorial requires more significant computational resources (e.g., a GPU, multiple CPS, or more than 30 seconds runtime), we suggest executing the notebook on your local computing resources and committing the `.ipynb` (with output cells) directly to the repository. Both types of tutorial formats are described in more detail below.

#### Small(ish) tutorials requiring only a CPU

Like with the [exoplanet](https://docs.exoplanet.codes/en/stable/user/dev/) codebase, MPoL tutorials are written as [MyST-NB `.md` source files](https://myst-nb.readthedocs.io/en/latest/authoring/basics.html#text-based-notebooks). During the Sphinx build process, they are converted to Jupyter notebooks and executed using the MyST build process. For these smaller tutorials, the `.md` file you create is the only thing you need to commit to the github repo (don't commit the `.ipynb` file to the git repository in this case). This practice keeps the git diffs small while making it easier to edit tutorials with an integrated development environment.

To write a tutorial, we suggest copying and rename one of the existing `.md` files in `docs/ci-tutorials/` to `docs/ci-tutorials/your-new-tutorial`, being sure to keep the header metadata. Then, you can write your tutorial using the [MyST markdown syntax](https://myst-nb.readthedocs.io/en/latest/authoring/text-notebooks.html).

When done, add a reference to your tutorial in the table of contents in `docs/index.rst`. E.g., if your contribution is the `ci-tutorials/gridder.md` file, add a `ci-tutorials/gridder.md` line. Then, you should be able to build the documentation as normal (i.e., `make html`) and your tutorial will be included in the documentation. The MyST-NB plugin is doing the work behind the scenes.

If your tutorial requires any extra build dependencies, please add them to the `EXTRA_REQUIRES['docs']` list in `setup.py`.

#### Larger tutorials requiring substantial computational resources

Radio interferometric datasets are frequently large, and sometimes realistic tutorials with real data require substantial computational resources beyond those provided in github workflows. Though more burdensome to package, these "end-to-end" tutorials are often the most useful for users.

Larger tutorials are not contributed in a continuously-integrated fashion, but instead are built and executed using local computational resources (these could be your laptop or a university research cluster). We still recommend that you write your tutorial as a `.md` file like the smaller tutorials, and update the `large-tutorials/Makefile` with your new filenames. The difference here is that you will build the `.ipynb` version of your tutorial locally using [Jupytext](https://jupytext.readthedocs.io/en/latest/) (you could use `make all` inside the `large-tutorials` directory, or just run `jupytext --to ipynb --execute my_tutorial.py`) you *will* want to commit the `.ipynb` file containing the cell output directly to the git repository.

The expectation is that these tutorials will only be rerun when the tutorial is updated, so the git diff issue is not as large a concern as it was with the continuously-integrated smaller tutorials. Like before, the MyST-NB plugin will see a Jupyter notebook and incorporate it during the build process. Because the larger tutorials are not continuously integrated, however, there is some concern that the codebase could diverge from that used to generate the tutorial, rendering the tutorial stale. We believe this risk is acceptable given the benefit that these larger tutorials provide and we intend to check the tutorials for staleness with at least every minor release.

To summarize, to write a large tutorial:

1. copy and rename one of the existing `.md` files in `docs/large-tutorials` folder
2. use [Jupytext](https://jupytext.readthedocs.io/en/latest/) to convert the `.md` to an `.ipynb` file and execute it on your local resources. You may want to add a line to the `docs/large-tutorials/Makefile` for your tutorial
3. commit the `.ipynb` to the MPoL repository, if you haven't already (i.e., you'll be committing *both* the `.md` and `.ipynb` files in this case).
4. if you run your notebook is run on a cluster, please also commit your submission script (e.g., SLURM, torque, moab). You may also consider additionally pasting the contents of the build script as a text cell inside the `.ipynb` for reference.

When done, add a reference to your tutorial in the documentation table of contents. E.g., if your contribution is the `large-tutorials/gpus.ipynb` file, add a `large-tutorials/gpus.ipynb` line to the table of contents.

#### Tutorial best practices

Tutorials should be self-contained. If the tutorial requires a dataset, the dataset should be publicly available and downloaded at the beginning of the script. If the dataset requires significant preprocessing (e.g., some multi-configuration ALMA datasets), those preprocessing steps should be in the tutorial. If the steps are tedious, one solution is to upload a preprocessed dataset to Zenodo and have the tutorial download the data product from there (the preprocessing scripts/steps should still be documented in the Zenodo repo and/or as part of the [mpoldatasets repository](https://github.com/MPoL-dev/mpoldatasets)). The guiding principle is that other developers should be able to successfully build the tutorial from start to finish without relying on any locally provided resources or datasets.

If you're thinking about contributing a tutorial and would like guidance on form or scope, please raise an [issue](https://github.com/MPoL-dev/MPoL/issues) or [discussion](https://github.com/MPoL-dev/MPoL/discussions) on the github repository.

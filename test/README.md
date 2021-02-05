# Testing for MPoL

Testing is carried out with `pytest`. Routines for testing the core `MPoL` functionality are included with this package. For more complicated workflows, additional tests may be implemented in outside packages.

You can install the package dependencies for testing via

    $ pip install .[test]

after you've cloned the repository and changed to the root of the repository (`MPoL`). This installs the extra packages required for testing (they are listed in `setup.py`).

To run all of the tests, from  the root of the repository, invoke

    $ python -m pytest

## Test cache

Several of the tests require mock data that is not practical to package within the github repository itself. These files are stored on Zenodo, and for continuous integration tests (e.g., on github workflows), they are downloaded as needed. 

However, local developers might need to run these tests frequently in the course of making changes to the package. If you are writing a test, the structure we are following is to keep all large files in a cache directory. Before testing, set the environment variable `MPOL_CACHE_DIR` to a location of your choosing and after the first download, subsequent tests will load files from this directory. If this environment variable is unset, then all test data files will be downloaded to a temporary directory which will eventually be cleaned up by the operating system. This means the test data files will be re-downloaded on each run of the tests. Depending on how fast your internet connection is, this might be a burden.


## Viewing plots

Some tests produce temporary files, like plots, that could be useful to view for development or debugging. Normally these are produced to a temporary directory created by the system which will be cleaned up after the tests finish. To preserve them, first create a plot directory and then run the tests with this `basetemp` specified 
    
    $ mkdir plotsdir
    $ python -m pytest --basetemp=plotsdir

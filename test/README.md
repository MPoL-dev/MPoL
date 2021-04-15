# Testing for MPoL

Testing is carried out with `pytest`. Routines for testing the core `MPoL` functionality are included with this package. For more complicated workflows, additional tests may be implemented in outside packages.

You can install the package dependencies for testing via

    $ pip install .[test]

after you've cloned the repository and changed to the root of the repository (`MPoL`). This installs the extra packages required for testing (they are listed in `setup.py`).

To run all of the tests, from  the root of the repository, invoke

    $ python -m pytest

## Test cache

Several of the tests require mock data that is not practical to package within the github repository itself. These files are stored on Zenodo, and for continuous integration tests (e.g., on github workflows), they are downloaded as needed using astropy cache utilities.


## Viewing plots

Some tests produce temporary files, like plots, that could be useful to view for development or debugging. Normally these are produced to a temporary directory created by the system which will be cleaned up after the tests finish. To preserve them, first create a plot directory and then run the tests with this `basetemp` specified 
    
    $ mkdir plotsdir
    $ python -m pytest --basetemp=plotsdir

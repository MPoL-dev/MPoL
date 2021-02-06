=======================
Developer Documentation 
=======================

If you are new to contributing code to a project, we recommend a quick read through the excellent contributing guides of the `exoplanet <https://docs.exoplanet.codes/en/stable/user/dev/>`_ and `astropy <https://docs.astropy.org/en/stable/development/workflow/development_workflow.html>`_ packages. What follows in this guide draws upon many of the suggestions from those two resources.

Code Bugs and Enhancements
--------------------------

If you find an issue with the code, documentation, or would like to make a suggestion for an improvement, please raise an issue on the `Github repository <https://github.com/MPoL-dev/MPoL/issues>`_.

   .. _testing-reference-label:

Testing
-------

MPoL includes a test suite written using `pytest <https://docs.pytest.org/>`_. You can install the package dependencies for testing via ::

    $ pip install .[test]

after you've cloned the repository and changed to the root of the repository. 

To run all of the tests, from  the root of the repository, invoke ::

    $ python -m pytest

If a test errors on a stable branch, please report what went wrong as an issue on the `Github repository <https://github.com/MPoL-dev/MPoL/issues>`_.

Test cache
==========

Several of the tests require mock data that is not practical to package within the github repository itself, and so it is stored on Zenodo. For continuous integration tests (e.g., on github workflows), mock data is downloaded with each run of the tests. Because the data is saved to a temporary system directory, it is cleaned up after each run.

Frequent downloading of moderately sized files can be burdensome, especially for developers who might want to run tests frequently in the course of making changes to the package. If you want to cache downloaded files, set your shell environment variable ``MPOL_CACHE_DIR`` to a location of your choosing. After the files are downloaded during the first test, subsequent tests will load files from this directory. 


Viewing test and debug plots
============================

Some tests produce temporary files, like plots, that could be useful to view for development or debugging. Normally these are saved to a temporary directory created by the system which will be cleaned up after the tests finish. To preserve them, first create a plot directory (e.g., ``plotsdir``) and then run the tests with this ``--basetemp`` specified ::
    
    $ mkdir plotsdir
    $ python -m pytest --basetemp=plotsdir


Contributing 
------------

We use a "fork and pull request" model for collaborative development on Github, using git. If you are unfamiliar with this workflow, check out this short Github guide on `forking projects <https://guides.github.com/activities/forking/>`_. For even more reference material, see the official Github documentation on `collaborating with issues and pull requests <https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests>`_.

After forking the MPoL repository to your own github account, clone the repository and install it with the development tools::

    $ git clone https://github.com/MPoL-dev/MPoL.git
    $ cd MPoL
    $ pip install .[dev]

.. note:: 

    If you use the zsh shell, you might need to try ``$ pip install '.[dev]'`` instead.

Before starting to make any changes, it's a good idea to first run the tests as described :ref:`testing-reference-label` to make sure everything passes.

Once you're done making your changes, run the test suite again. If any tests fail, please address these issues before submitting a pull request. If you've changed a substantial amount of code or added new features, please write new tests covering these changes. Instructions on how to write new tests can be found in the `pytest documentation <https://docs.pytest.org/en/stable/contents.html#toc>`_ and the ``MPoL/tests/`` directory contains many examples. 

Once you are satisfied with your changes and all tests pass, initiate a pull request back to the main `MPoL-dev/MPoL repository <https://github.com/MPoL-dev/MPoL/>`_ as described in the Github guide on `forking projects <https://guides.github.com/activities/forking/>`_. Thank you for your contribution!

If you find that you're making regular contributions to the package, consider contacting `Ian Czekala <https://sites.psu.edu/iczekala/>`_ to become a member of the MPoL-dev organization on Github.


Contributing tutorials
----------------------

Like with the `exoplanet <https://docs.exoplanet.codes/en/stable/user/dev/>`_ codebase, MPoL tutorials are written as ``.py`` python files and converted to Jupyter notebooks using `jupytex <https://jupytext.readthedocs.io/en/latest/>`_. This practice prevents the MPoL github repository from growing too with bloated Jupyter notebooks filled with evaluated cells.

To contribute a new tutorial, we would recommend starting by

1. copying one of the existing `.py` files in `docs/tutorials/`
2. starting a jupyter notebook kernel
3. opening the `.py` file as a notebook and editing it like you would any other Jupyter notebook. If you've already installed the `jupytex <https://jupytext.readthedocs.io/en/latest/>`_. tool (as part of ``pip install .[dev]``), your changes in the Jupyter notebook window should be automatically saved back to the ``.py`` file.

You can learn more about this neat plugin on the `jupytex <https://jupytext.readthedocs.io/en/latest/>`_ page. You don't need to worry about pairing notebooks---we're only interested in keeping the ``.py`` file up to date and committed to source control. The ``.py`` file you created is the only thing you need to commit to the github repo (don't commit ``.ipynb`` files to the git repository).


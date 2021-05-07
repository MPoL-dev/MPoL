.. _developer-documentation-label:

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

.. note::

    If you use the zsh shell, you might need to try ``$ pip install -e '.[test]'`` instead.


To run all of the tests, from  the root of the repository, invoke ::

    $ python -m pytest

If a test errors on a stable branch, please report what went wrong as an issue on the `Github repository <https://github.com/MPoL-dev/MPoL/issues>`_.

Test cache
==========

Several of the tests require mock data that is not practical to package within the github repository itself, and so it is stored on Zenodo and downloaded using astropy caching functions. If you run into trouble with the test cache becoming stale, you can delete it by deleting the ``.mpol/cache`` folder in home directory.


Viewing test and debug plots
============================

Some tests produce temporary files, like plots, that could be useful to view for development or debugging. Normally these are saved to a temporary directory created by the system which will be cleaned up after the tests finish. To preserve them, first create a plot directory (e.g., ``plotsdir``) and then run the tests with this ``--basetemp`` specified ::

    $ mkdir plotsdir
    $ python -m pytest --basetemp=plotsdir


Contributing
------------

We use a "fork and pull request" model for collaborative development on Github, using git. If you are unfamiliar with this workflow, check out this short Github guide on `forking projects <https://guides.github.com/activities/forking/>`_. For even more reference material, see the official Github documentation on `collaborating with issues and pull requests <https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests>`_.

After forking the MPoL repository to your own github account, clone the repository and install it with the development tools in editable mode::

    $ git clone https://github.com/MPoL-dev/MPoL.git
    $ cd MPoL
    $ pip install -e .[dev]

.. note::

    If you use the zsh shell, you might need to try ``$ pip install -e '.[dev]'`` instead.

Before starting to make any changes, you should first run the tests as described :ref:`testing-reference-label` to make sure everything passes.

Once you're done making your changes, run the test suite again. If any tests fail, please address these issues before submitting a pull request. If you've changed a substantial amount of code or added new features, you should write new tests covering these changes. Instructions on how to write new tests can be found in the `pytest documentation <https://docs.pytest.org/en/stable/contents.html#toc>`_ and the ``MPoL/tests/`` directory contains many examples.

Once you are satisfied with your changes and all tests pass, initiate a pull request back to the main `MPoL-dev/MPoL repository <https://github.com/MPoL-dev/MPoL/>`_ as described in the Github guide on `forking projects <https://guides.github.com/activities/forking/>`_. Thank you for your contribution!


Developing tutorials
====================

Like with the `exoplanet <https://docs.exoplanet.codes/en/stable/user/dev/>`_ codebase, MPoL tutorials are written as ``.py`` python files and converted to Jupyter notebooks using `jupytext <https://jupytext.readthedocs.io/en/latest/>`_. You can learn more about this neat plugin on the `jupytext <https://jupytext.readthedocs.io/en/latest/>`_ page. You don't need to worry about pairing notebooks---we're only interested in keeping the ``.py`` file up to date and committed to source control. For small tutorials, the ``.py`` file you create is the only thing you need to commit to the github repo (don't commit the ``.ipynb`` file to the git repository in this case). This practice keeps the git diffs small while making it easier to edit tutorials with an integrated development environment.

To write a tutorial:

1. copy and rename one of the existing ``.py`` files in ``docs/tutorials/`` to ``docs/tutorials/your-new-tutorial``, being sure to keep the header metadata
2. start a Jupyter notebook kernel
3. open the ``.py`` file as a notebook and edit it like you would any other Jupyter notebook. If you've already installed the `jupytext <https://jupytext.readthedocs.io/en/latest/>`_ tool (as part of ``pip install -e .[dev]``), your changes in the Jupyter notebook window should be automatically saved back to the ``.py`` file. As you progress, make sure you commit your changes in the ``.py`` file back to the repository (but don't commit the ``.ipynb`` file).

When done, add a reference to your tutorial in the documentation table of contents. E.g., if your contribution is the ``tutorials/gridder.py`` file, add a ``tutorials/gridder`` line to the tabel of contents.

To build the docs locally, ``cd`` to the docs folder and run ``make html``. This script includes the line ::

    jupytext --to ipynb --execute tutorials/*.py

which converts your ``.py`` file to a ``.ipynb`` file and executes its contents, storing the cell output to the notebook. Then, when Sphinx builds the documention, the `nbsphinx <https://nbsphinx.readthedocs.io/>`_ plugin sees a Jupyter notebook and incorporates it into the build. If you've added any extra documentation build dependencies, please add them to the ``EXTRA_REQUIRES['docs']`` list in ``setup.py``.

Tutorial best practices
=======================

Tutorials should still be self-contained. If the tutorial requires a dataset, the dataset should be publically available and downloaded at the beginning of the script. If the dataset requires significant preprocessing (e.g., some multi-configuration ALMA datasets), those preprocessing steps should be in the tutorial. If the steps are tedious, one solution is to upload a preprocessed dataset to Zenodo and have the tutorial download the data product from there (the preprocessing scripts/steps should still be documented in the Zenodo repo). The guiding principle is that other developers should be able to successfully build the tutorial from start to finish without relying on any locally provided resources or datasets.

If you're thinking about contributing a tutorial and would like guidance on form or scope, please raise an `issue <https://github.com/MPoL-dev/MPoL/issues>`_ or `discussion <https://github.com/MPoL-dev/MPoL/discussions>`_ on the github repository.

import os

# -- Project information -----------------------------------------------------
from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution("MPoL").version
except DistributionNotFound:
    __version__ = "unknown version"

project = "MPoL"
copyright = "2019-21, Ian Czekala"
author = "Ian Czekala"

# The full version, including alpha/beta/rc tags
version = __version__
release = __version__

# -- General configuration ---------------------------------------------------
extensions = [
    "faculty_sphinx_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "nbsphinx",
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
]

autodoc_mock_imports = ["torch", "torchvision"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# nbsphinx configuration
suppress_warnings = [
    "nbsphinx",
]

nbsphinx_execute_arguments = [
    "--InlineBackend.figure_format='retina'",
    # "--InlineBackend.figure_format={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 200}",
]

# -- Options for HTML output -------------------------------------------------
html_theme = "faculty-sphinx-theme"
html_theme_options = {"analytics_id": "UA-5472810-8"}

html_logo = "logo.png"
html_favicon = "favicon.ico"

master_doc = "index"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Mermaid configuration
# only works locally
mermaid_output_format = "svg"

if os.getenv("CI"):
    # if True, we're running on github actions and need
    # to use the path of the installed mmdc
    # relative to docs/ directory!
    # (mmdc itself not in $PATH automatically, like local)
    mermaid_cmd = "../node_modules/.bin/mmdc"

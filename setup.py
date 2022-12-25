import os.path

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


version = get_version("src/mpol/__init__.py")


EXTRA_REQUIRES = {
    "test": [
        "pytest",
        "pytest-cov",
        "matplotlib",
        "requests",
        "astropy",
        "tensorboard",
    ],
    "docs": [
        "sphinx>=2.3.0",
        "numpy",
        "jupytext",
        "ipython!=8.7.0", # broken version for syntax higlight https://github.com/spatialaudio/nbsphinx/issues/687
        "nbsphinx",
        "sphinx_book_theme",
        "sphinx_copybutton",
        "jupyter",
        "nbconvert",
        "matplotlib",
        "sphinxcontrib-mermaid",
        "astropy",
        "tensorboard",
        "myst-nb",
        "jupyter-cache",
    ],
}

EXTRA_REQUIRES["dev"] = (
    EXTRA_REQUIRES["test"] + EXTRA_REQUIRES["docs"] + ["pylint", "black", "pre-commit"]
)


setuptools.setup(
    name="MPoL",
    version=version,
    author="Ian Czekala",
    author_email="iancze@gmail.com",
    description="Regularized Maximum Likelihood Imaging for Radio Astronomy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/iancze/MPoL",
    install_requires=["numpy", "scipy", "torch>=1.8.0", "torchvision", "torchaudio", "torchkbnufft", "astropy"],
    extras_require=EXTRA_REQUIRES,
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)

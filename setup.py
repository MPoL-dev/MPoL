import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

# it's not that mpol is installed at this point, just that 
# it's in the current working directory and we can actually
# import just the __init__.py file, which contains __version__
import mpol
version=mpol.__version__


setuptools.setup(
    name="MPoL",
    version=version,
    author="Ian Czekala",
    author_email="iancze@gmail.com",
    description="Regularized Maximum Likelihood Imaging for Radio Astronomy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/iancze/MPoL",
    install_requires=["numpy", "scipy", "torch", "torchvision"],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.5",
)

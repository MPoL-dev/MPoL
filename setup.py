import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="MPoL",  # Replace with your own username
    version="0.0.1",
    author="Ian Czekala",
    author_email="iancze@gmail.com",
    description="Maximum Entropy Imaging for Radio Astronomy",
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

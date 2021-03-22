#!/bin/bash

# build the notebooks
jupytext --to ipynb --execute ci-tutorials/*.py

# do the sphinx build
python -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html

# create PNGs for fun/presentations
mmdc -i _static/mmd/SimpleNet.mmd -o _build/html/_images/SimpleNet.svg


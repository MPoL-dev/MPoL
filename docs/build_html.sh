#!/bin/bash

# create SVGs for fun/presentations
mmdc -i _static/mmd/src/SimpleNet.mmd -o _static/mmd/build/SimpleNet.svg

# build the notebooks
# jupytext --to ipynb --execute ci-tutorials/*.py
jupytext --to ipynb --execute ci-tutorials/gridder.py
jupytext --to ipynb --execute ci-tutorials/optimization.py

# do the sphinx build
python -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html


# # + nbsphinx="hidden"
# from IPython.display import SVG, Image, display

# display(SVG(filename="../_static/mmd/SimpleNet.svg"))
# # -

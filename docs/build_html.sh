#!/bin/bash

# build the notebooks
jupytext --to ipynb --execute tutorials/*.py
python -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
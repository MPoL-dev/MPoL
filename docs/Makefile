# makefile for Sphinx documentation

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile html clean

CI-NOTEBOOKS := ci-tutorials/PyTorch.ipynb ci-tutorials/gridder.ipynb ci-tutorials/optimization.ipynb ci-tutorials/crossvalidation.ipynb ci-tutorials/initializedirtyimage.ipynb
CHARTS := _static/mmd/build/SimpleNet.svg _static/mmd/build/ImageCube.svg _static/mmd/build/BaseCube.svg _static/mmd/build/SkyModel.svg
clean:
	rm -rf _build
	# rm -rf ${CI-NOTEBOOKS}
	rm -rf ci-tutorials/.ipynb_checkpoints
	rm -rf ci-tutorials/runs
	rm -rf ${CHARTS}
	rm -rf _static/baselines/build/baselines.csv

# ci-tutorials/%.ipynb: ci-tutorials/%.py ${CHARTS}
	# jupytext --to ipynb --execute $<

# baseline table
_static/baselines/build/baselines.csv: _static/baselines/src/print_conversions.py
	mkdir -p _static/baselines/build
	python _static/baselines/src/print_conversions.py $@

# fftshift figure
_static/fftshift/build/plot.png: _static/fftshift/src/plot.py
	mkdir -p _static/fftshift/build
	python _static/fftshift/src/plot.py $@

# mermaid.js files
_static/mmd/build/%.svg: _static/mmd/src/%.mmd
	mkdir -p _static/mmd/build
	${MERMAID_PATH}mmdc -i $^ -o $@

html: ${CHARTS} _static/baselines/build/baselines.csv _static/fftshift/build/plot.png
	python -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html

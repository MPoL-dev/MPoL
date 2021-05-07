# makefile for Sphinx documentation
#

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

NOTEBOOKS := tutorials/gridder.ipynb tutorials/optimization.ipynb tutorials/crossvalidation.ipynb
CHARTS := _static/mmd/build/SimpleNet.svg _static/mmd/build/ImageCube.svg _static/mmd/build/BaseCube.svg _static/mmd/build/SkyModel.svg

clean:
	rm -rf _build
	rm -rf ${NOTEBOOKS}
	rm -rf tutorials/.ipynb_checkpoints
	rm -rf tutorials/runs
	rm -rf ${CHARTS}
	rm -rf _static/baselines/build/baselines.csv

tutorials/%.ipynb: tutorials/%.py ${CHARTS}
	jupytext --to ipynb --execute $<

# baseline table
_static/baselines/build/baselines.csv:
	mkdir -p _static/baselines/build
	python _static/baselines/src/print_conversions.py $@

# mermaid.js files
_static/mmd/build/%.svg: _static/mmd/src/%.mmd
	mkdir -p _static/mmd/build
	${MERMAID_PATH}mmdc -i $^ -o $@

html: ${NOTEBOOKS} ${CHARTS} _static/baselines/build/baselines.csv
	python -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html

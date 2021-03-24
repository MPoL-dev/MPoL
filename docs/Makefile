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

NOTEBOOKS := ci-tutorials/gridder.ipynb ci-tutorials/optimization.ipynb
CHARTS := _static/mmd/build/SimpleNet.svg _static/mmd/build/ImageCube.svg _static/mmd/build/BaseCube.svg _static/mmd/build/SkyModel.svg

clean:
	rm -rf _build 
	rm -rf ${NOTEBOOKS}
	rm -rf ci-tutorials/.ipynb_checkpoints
	rm -rf ${CHARTS}


ci-tutorials/%.ipynb: ci-tutorials/%.py ${CHARTS}
	jupytext --to ipynb --execute $<

# mermaid.js files
_static/mmd/build/%.svg: _static/mmd/src/%.mmd
	${MERMAID_PATH}mmdc -i $^ -o $@

html: ${NOTEBOOKS} ${CHARTS}
	python -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
# %: Makefile
# 	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

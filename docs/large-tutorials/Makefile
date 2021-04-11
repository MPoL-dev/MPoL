
.PHONY: Makefile clean all 

all : crossvalidation.ipynb

%.ipynb: %.py notebook_setup.py
	jupytext --to ipynb --execute $<

clean:
	rm -rf runs
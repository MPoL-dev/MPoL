LARGE-NOTEBOOKS := HD143006_part_1.ipynb HD143006_part_2.ipynb pyro.ipynb

%.ipynb: %.md
	jupytext --to ipynb --execute $<


all: ${LARGE-NOTEBOOKS}

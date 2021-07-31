LARGE-NOTEBOOKS := HD143006_part_1.ipynb HD143006_part_2.ipynb

%.ipynb: %.py
	jupytext --to ipynb --execute $<


all: ${LARGE-NOTEBOOKS}

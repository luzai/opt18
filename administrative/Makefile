SHELL = /bin/bash

main:
	# while [ 1 -gt 0  ] ; do echo hello ; latexmk -xelatex -interaction=nonstopmode -halt-on-error main ; sleep 3; done ; 
	# make clean	
	xelatex -interaction=nonstopmode -halt-on-error main
	# bibtex main
	# pdflatex -interaction=nonstopmode -halt-on-error main
	# pdflatex -interaction=nonstopmode -halt-on-error main

clean:
	git clean -dfX

watch: main.tex 
	fswatch -o $^ | xargs -n1 -I{} make
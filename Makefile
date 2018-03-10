SHELL = /bin/bash

main.pdf: main.tex clean
	xelatex -interaction=nonstopmode -halt-on-error main
	xelatex -interaction=nonstopmode -halt-on-error main
	# biber main
main.tex: main.md 
	# pandoc -s -S -t beamer -F pandoc-fignos -F pandoc-citeproc --template main_template.tex --slide-level=3 main.md -o main.tex 
	pandoc -s -S -t beamer --template main_template.tex --slide-level=3 main.md -o main.tex --listings --highlight-style=pygments 
plain:
	# make clean	
	# pdflatex -interaction=nonstopmode -halt-on-error main
	xelatex -interaction=nonstopmode -halt-on-error main
	xelatex -interaction=nonstopmode -halt-on-error main
	# bibtex main
	# pdflatex -interaction=nonstopmode -halt-on-error main
	# pdflatex -interaction=nonstopmode -halt-on-error main

clean:
	git clean -dfX
	# rm -rf main.tex main.nav main.toc

watch: main.tex 
	fswatch -o $^ | xargs -n1 -I{} make

forever: 
	while [ 1 -gt 0  ] ; do echo hello ; latexmk -xelatex -interaction=nonstopmode -halt-on-error main ; sleep 3; done ; 

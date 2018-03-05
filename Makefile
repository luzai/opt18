main:
	# make clean
	latexmk -xelatex -interaction=nonstopmode -halt-on-error main 
	# xelatex -interaction=nonstopmode -halt-on-error main
	# bibtex main
	# pdflatex -interaction=nonstopmode -halt-on-error main
	# pdflatex -interaction=nonstopmode -halt-on-error main

clean:
	git clean -dfX

watch: main.tex 
	fswatch -o $^ | xargs -n1 -I{} make
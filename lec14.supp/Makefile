SHELL = /bin/bash

main: 
	xelatex --interaction=nonstopmode -halt-on-error main
	cp main.pdf lec14_conv.supp.pdf
	cp lec14_conv.supp.pdf ../opt-release/	

clean:
	# git clean -dfX
	rm -rf *.nav *.toc *.aux *.log *.snm *.out *.fls *.fbd* *.pdf

watch: main.tex 
	fswatch -o $^ | xargs -n1 -I{} make

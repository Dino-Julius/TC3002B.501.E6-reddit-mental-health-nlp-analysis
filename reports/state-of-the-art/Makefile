.PHONY: all clean zip

all:
	mkdir -p build
	latexmk -pdf -interaction=nonstopmode -synctex=1 -outdir=build main.tex

clean:
	rm -rf build

zip:
	rm -f TC3002B.501.E6-StateOfTheArt-LaTeX.zip
	zip -r TC3002B.501.E6-StateOfTheArt-LaTeX.zip main.tex references.bib sections tables figures Makefile .latexmkrc README.md

# Build the VLM4RWD 2026 workshop paper.
# Requires tectonic (brew install tectonic); it runs bibtex/reruns automatically.
.PHONY: all clean watch

all: main.pdf

main.pdf: main.tex references.bib neurips_2026.sty
	tectonic main.tex

watch:
	tectonic -X watch main.tex || tectonic main.tex

clean:
	rm -f main.pdf main.aux main.bbl main.blg main.log main.out

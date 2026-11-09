# AAAI-27 (GSR-VLM workshop) build. aaai2027.sty requires pdfTeX, so this uses
# pdflatex + bibtex (brew install texlive), not tectonic.
.PHONY: all
all: main.pdf
main.pdf: main.tex references.bib aaai2027.sty aaai2027.bst $(wildcard tables/*.tex tables/legacy/*.tex figures/*)
	pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
	bibtex main >/dev/null
	pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
	pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
	@grep -E "Citation .* undefined|Reference .* undefined" main.log || true
	@echo "built main.pdf ($$(pdfinfo main.pdf | awk '/^Pages/{print $$2}') pages)"

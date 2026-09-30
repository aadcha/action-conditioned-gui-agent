# Build the VLM4RWD 2026 workshop paper.
# Requires tectonic (brew install tectonic); it runs bibtex/reruns automatically.
#
#   make            -> main.pdf      double-blind submission form (line numbers,
#                                    anonymous author block, "Do not distribute")
#   make preprint   -> preprint.pdf  de-anonymized arXiv form, generated from
#                                    main.tex by make_preprint.py
#
# main.tex is the single source of truth. preprint.tex is a build artifact;
# edit main.tex and re-run `make preprint`.
.PHONY: all preprint clean watch

all: main.pdf

main.pdf: main.tex references.bib neurips_2026.sty
	tectonic main.tex

preprint: preprint.pdf

preprint.tex: main.tex make_preprint.py
	python3 make_preprint.py

preprint.pdf: preprint.tex references.bib neurips_2026.sty
	tectonic preprint.tex

watch:
	tectonic -X watch main.tex || tectonic main.tex

clean:
	rm -f main.pdf main.aux main.bbl main.blg main.log main.out
	rm -f preprint.pdf preprint.tex preprint.aux preprint.bbl preprint.blg preprint.log preprint.out

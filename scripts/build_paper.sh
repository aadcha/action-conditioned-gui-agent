#!/bin/bash
# Build both paper versions from the shared body, plus an arXiv source tarball:
#   overleaf_submission/paper.pdf          arXiv build (CVPR two-column style)
#   overleaf_submission/paper_neurips.pdf  workshop build (NeurIPS 2026 style,
#                                          submission mode: anonymized + line numbers)
#   dist/arxiv_src.tar.gz                  ready-to-upload arXiv source bundle
set -e
cd "$(dirname "$0")/.."
uv run python scripts/p10_render_paper_tables.py
cp paper.tex body.tex paper_neurips.tex overleaf_submission/
rm -rf overleaf_submission/tables && cp -r tables overleaf_submission/tables
cd overleaf_submission
tectonic --keep-intermediates paper.tex
tectonic paper_neurips.tex
# arXiv bundle: sources + .bbl (arXiv compiles TeX but not bibtex-with-custom-bst reliably)
mkdir -p ../dist/arxiv_src/tables ../dist/arxiv_src/figures
cp paper.tex body.tex paper.bbl paper.bib cvpr.sty cvpr_eso.sty eso-pic.sty ieee.bst ../dist/arxiv_src/
cp tables/*.tex ../dist/arxiv_src/tables/
for f in stage1_mind2web_vs_aitw.png ablation_ABCD_all_with_coords.png \
         compounding_error_per_class.png scaling_curve.png attn_example_1.png; do
  cp "figures/$f" ../dist/arxiv_src/figures/
done
cd ../dist && tar czf arxiv_src.tar.gz arxiv_src && rm -rf arxiv_src && cd ..
rm -f overleaf_submission/paper.aux overleaf_submission/paper.log overleaf_submission/paper.blg
echo "built: overleaf_submission/paper.pdf, paper_neurips.pdf, dist/arxiv_src.tar.gz"

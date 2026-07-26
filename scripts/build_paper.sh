#!/bin/bash
# Build both paper versions from the shared body:
#   overleaf_submission/paper.pdf          arXiv build (CVPR two-column style)
#   overleaf_submission/paper_neurips.pdf  workshop build (NeurIPS 2026 style,
#                                          submission mode: anonymized + line numbers)
set -e
cd "$(dirname "$0")/.."
uv run python scripts/p10_render_paper_tables.py
cp paper.tex body.tex paper_neurips.tex overleaf_submission/
rm -rf overleaf_submission/tables && cp -r tables overleaf_submission/tables
cd overleaf_submission
tectonic paper.tex
tectonic paper_neurips.tex
echo "built overleaf_submission/paper.pdf and paper_neurips.pdf"

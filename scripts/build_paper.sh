#!/bin/bash
# Build the arXiv (CVPR-style) paper: sync source + generated tables into
# overleaf_submission/ and compile with tectonic.
set -e
cd "$(dirname "$0")/.."
uv run python scripts/p10_render_paper_tables.py
cp paper.tex overleaf_submission/paper.tex
rm -rf overleaf_submission/tables && cp -r tables overleaf_submission/tables
cd overleaf_submission && tectonic paper.tex
echo "built overleaf_submission/paper.pdf"

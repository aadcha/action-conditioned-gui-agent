#!/usr/bin/env bash
# Pull every Phase 8 artifact from the stage1-cache Volume and rebuild the
# consolidated tables/figures. Safe to re-run; each step is idempotent.
set -euo pipefail
cd "$(dirname "$0")/.."
export MODAL_PROFILE="${MODAL_PROFILE:-sentinel}"

echo "== stage2 run JSONs -> results/phase4/"
uv run modal run modal_app.py::list_stage2_runs | grep -E "found|saved" | tail -3

echo "== attention aggregate -> results/phase8/attn_aggregate/"
uv run modal run modal_app.py::pull_attn_aggregate | tail -2

echo "== qualitative_v2 -> results/phase8/qualitative_v2/"
mkdir -p results/phase8/qualitative_v2
uv run modal volume get --force stage1-cache qualitative_v2 results/phase8/ 2>&1 | tail -2 || true

echo "== consolidate"
uv run python scripts/p8_consolidate.py > /dev/null && echo "wrote results/phase8/PHASE8_RESULTS.md"
if ls results/phase8/attn_aggregate/*/attn_aggregate_*.json >/dev/null 2>&1; then
  uv run python scripts/p8_attn_summary.py > /dev/null && echo "wrote results/phase8/ATTN_AGGREGATE.md"
fi
if [ -f results/phase8/qualitative_v2/qualitative_grounding.png ]; then
  cp results/phase8/qualitative_v2/qualitative_grounding.png neurips2026/figures/qualitative_grounding_v2.png
  echo "copied qualitative_grounding_v2.png into neurips2026/figures/"
fi

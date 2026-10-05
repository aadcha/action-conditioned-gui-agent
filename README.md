# Action-Type-Conditioned Grounding for GUI Agents

Code, data and paper source for *Decoupling What from Where: How Should a Small
GUI Grounding Model Receive the Action Type?*

A GUI agent has to decide which action to take and where on the screen to take
it. Native agent models emit both in one autoregressive stream. This repository
factors them apart and asks how the grounding half should receive the action
type: a frozen Qwen2-VL-2B feeding an MLP predicts the type, and a LoRA-tuned
Qwen2-VL-2B grounds the action to a coordinate string conditioned on it.

Six ways of supplying the type are compared under matched data, compute,
adapters and decoding: a flat baseline, an auxiliary classification loss, a
hard-routed action word, an additive learned embedding, a prepended learned
token, and the type written into the prompt.

The paper is in `neurips2026/`. Build it with `make`, or `make preprint` for the
de-anonymized version.

## Setup

```bash
uv sync --extra dev
uv run pytest tests/
```

Training runs on Modal:

```bash
uv sync --extra modal
modal token new
modal run modal_app.py::smoke          # about 5 minutes on an L4
```

The first run pulls Qwen2-VL-2B (4.4 GB) into a Modal Volume; later cold starts
skip the download.

The 2B base is small enough to load on a 24 GB Mac via MPS, which is slow but
useful for debugging without spending credits:

```bash
uv run python scripts/smoke_test.py
```

This confirms the model loads, the LoRA adapter attaches, and a forward plus
generate completes. It synthesizes a placeholder screenshot unless given
`--image path/to/screenshot.png`. If it runs out of memory, set `dtype` to
`float16` in `configs/smoke_test.yaml`.

On multi-GPU machines, `device_map="auto"` can split layers in ways that
interact badly with LoRA. If the model fits on one GPU, set
`device_map: {"": 0}` in the config.

## Reproducing the paper

Every number comes from run JSONs in `results/phase4/`, each carrying a
`per_example_dist` array over the shared validation slice.

```bash
uv run modal run modal_app.py::list_stage2_runs   # pull run JSONs from the Modal volume
uv run python scripts/p8_consolidate.py           # rebuild every table, figure and PHASE8_RESULTS.md
cd neurips2026 && make                            # the paper
```

| paper element | produced by |
|---|---|
| Tables 1, 3 (headline, control) | `section_variants` in `scripts/p8_consolidate.py` |
| Table 4 (interventions) | `section_interventions`, from `results/phase4/interv_*.json` |
| Table 7 (type events removed) | `section_notype`, from the `*_notype.json` runs |
| Table 9 (D-token learning-rate sweep) | `section_dtoken_lr`, from the `*_alr*.json` runs |
| Figure 2, Table 8 (scaling) | `section_scaling` |
| episode-cluster bootstrap | `_episode_bootstrap`, clusters from `results/phase8/val_episodes.json` |
| per-class labels | `results/phase8/qualitative_v2/render.json` |

Statistics live in `src/eval/bootstrap.py` and the `paired()` helper in
`scripts/p8_consolidate.py`, which reports a pooled-unit interval, an
example-cluster interval, the episode-cluster interval used in the paper, a
boundary-episode-excluded variant, and a seed-level paired t-test.

Consolidated results, including every contrast the paper does not have room
for, are in `results/phase8/PHASE8_RESULTS.md`.

## Repo layout

```
src/models/      Qwen2-VL + LoRA loaders, the Stage-2 grounding model
src/data/        canonical action taxonomy, AITW and Mind2Web loaders
src/train/       training loops, coordinate serialization, grounding eval
src/eval/        paired bootstrap and permutation tests
modal_app.py     every Modal entry point; one remote plus one local entrypoint per experiment
scripts/         consolidation, figures, analysis
results/         run JSONs and per-phase writeups
neurips2026/     paper source, tables, figures
tests/           29 tests, no GPU required
```

AITW screenshots in the `cjfcsjt/AITW_General` mirror are stored as raw RGB
bytes with no image header. `src/data/aitw._decode_aitw_image_bytes` handles the
seven observed resolutions; `PIL.Image.open` on the raw buffer will fail.

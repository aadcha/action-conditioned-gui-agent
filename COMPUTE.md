# Compute Budget & Plan

**Total cloud credits available: $750**

| Provider | Credits | Best use |
|---|---|---|
| Modal | $200 | Primary — serverless GPU, no idle cost, fast iteration |
| Google Cloud | $300 + $50 = $350 | Bulk training on spot A100s |
| AWS | $100 | Backup / overflow |
| Azure | $100 | Backup / overflow |

No on-prem cluster. Every GPU-second is metered.

---

## Per-GPU-hour reference (approximate, as of mid-2026)

| Provider | GPU | $/hr on-demand | $/hr spot | Notes |
|---|---|---|---|---|
| Modal | L4 (24 GB) | ~$0.80 | n/a | Cheap iteration; fits 2B comfortably |
| Modal | A100 40 GB | ~$2.10 | n/a | Real training for 2B; 7B with checkpointing |
| Modal | A100 80 GB | ~$3.40 | n/a | Comfortable for 7B |
| Modal | H100 80 GB | ~$4.00 | n/a | Overkill for 2B |
| GCP | L4 24 GB | ~$0.70 | ~$0.30 | Cheapest L4 |
| GCP | A100 40 GB | ~$3.67 | **~$1.50** | Spot is the workhorse |
| GCP | A100 80 GB | ~$5.07 | ~$2.00 | Only if 40 GB OOMs |
| AWS | A10G g5.xlarge | ~$1.00 | ~$0.40 | Marginal for 7B |
| Azure | A100 40 GB | ~$3.50 | ~$1.20 | Use as last resort |

---

## Workload cost estimates (Qwen2-VL-2B default)

Assumes batch size 4 with grad accumulation to effective 16–32, bf16, gradient checkpointing,
class-balanced sampling on a **~25k Mind2Web subset** (the chosen first-pass scope).

| Workload | Where | Est. wall time | Est. cost |
|---|---|---|---|
| Smoke test (cold start + 30 s run) | Modal L4 | ~5 min | ~$0.07 |
| Stage 1 classifier (MLP on cached features) | Modal L4 (1× pass to cache features) + CPU | ~1 hr GPU + minutes CPU | ~$1 |
| Stage 2 dev / debug | Modal L4 → A100 | ~30 hrs total | ~$40 |
| Ablation A (flat LoRA baseline) | GCP spot A100 40 GB | ~15 hrs | ~$23 |
| Ablation B (aux loss) | GCP spot A100 40 GB | ~15 hrs | ~$23 |
| Ablation C (hard routing — eval-time only, no retrain) | Modal L4 | ~2 hrs | ~$2 |
| Ablation D (ours: type-embedding) | GCP spot A100 40 GB | ~15 hrs | ~$23 |
| Eval reruns + diagnostics | Modal L4/A100 | ~20 hrs | ~$25 |
| **Stretch: Qwen2-VL-7B on variant D** | Modal A100 80 GB | ~40 hrs | ~$140 |
| **Subtotal (2B-only)** | | | **~$140** |
| **Subtotal (with 7B stretch)** | | | **~$280** |

Even with the 7B stretch run, **~$470 buffer remains** for failed runs, longer-than-expected
training, or additional ablations (LoRA rank sweep, sampler ablation, etc.).

---

## Allocation strategy

1. **Phase 2–3** (data pipeline + classifier): **Modal only**. ~$50 total. Iterate fast.
2. **Phase 4** (Stage 2 development): **Modal L4 → A100**. Switch to A100 only once training-loop is verified on L4 with a tiny subset.
3. **Phase 6** (the four ablations): **GCP spot A100 40 GB**. The bulk training spend.
4. **Phase 7** (eval, diagnostics, attention viz): **Modal L4**.
5. **Stretch 7B run**: **Modal A100 80 GB**, only if 2B ablation table is in hand and credits are healthy.

Hold AWS and Azure unused as emergency overflow.

---

## Hard rules to stay in budget

1. **Default to L4 / spot A100.** Never use on-demand A100s unless a deadline forces it.
2. **Checkpoint every N steps.** GCP spot can preempt at any time; losing 10 hrs of training is ~$15.
3. **Smoke test on Modal L4 before scaling up.** A bad config caught on L4 costs cents; the same bug on H100 costs dollars.
4. **Cache HuggingFace weights to a Modal Volume.** First Qwen2-VL-2B download is ~4.4 GB; subsequent cold starts skip it. See `modal_app.py` (`hf_cache`).
5. **Track spend per run.** Log `gpu_type`, `wall_seconds`, and estimated cost to W&B alongside loss curves.

---

## Running spend tally

| Date | Run | Provider / GPU | Wall hrs | Cost | Cumulative |
|---|---|---|---|---|---|
| _none yet_ | | | | | $0.00 |

Update this table after each significant run.

---

## Why Qwen2-VL-2B as the default

The original plan was 7B. Cost forced the switch. The architecture, processor, LoRA target
modules, and every ablation (A/B/C/D) are **identical** between 2B and 7B — the experiment
*delta* (the contribution of action-type conditioning) is what the paper claims, and that
delta is preserved at 2B. Absolute numbers will be lower; we note this transparently in the
writeup. A single 7B run on the winning variant goes in as the headline if credits allow.

"""Phase 5 — generate the headline ablation markdown from ablation_summary.json.

Idempotent. Re-run any time after `scripts/p5_ablation_table.py`.

Output: results/phase4/PHASE5_HEADLINE.md
"""

from __future__ import annotations

import json
import math
from pathlib import Path

PHASE4_DIR = Path(__file__).resolve().parent.parent / "results" / "phase4"
OUT = PHASE4_DIR / "PHASE5_HEADLINE.md"


def _fmt_pm(mean: float, std: float, n_seeds: int) -> str:
    if math.isnan(mean):
        return "n/a"
    if n_seeds <= 1:
        return f"{mean:.3f}"
    return f"{mean:.3f} ± {std:.3f}"


def _delta(a: dict, b: dict, metric: str) -> tuple[float, float, float]:
    am = a[metric]["mean"]; as_ = a[metric]["std"]
    bm = b[metric]["mean"]; bs = b[metric]["std"]
    d = bm - am
    pooled = math.sqrt(as_ ** 2 + bs ** 2)
    return d, pooled, abs(d) / pooled if pooled > 0 else float("inf")


def main() -> None:
    summary_path = PHASE4_DIR / "ablation_summary.json"
    if not summary_path.exists():
        print("[p5-render] ablation_summary.json missing — run p5_ablation_table.py first")
        return
    summary: dict = json.loads(summary_path.read_text())
    mixes = sorted({s["data_mix"] for s in summary.values()})

    lines: list[str] = []
    lines.append("# Phase 5 — Ablation headline (Variant A vs Variant D)")
    lines.append("")
    lines.append("Matched-compute comparison of the project's central architectural")
    lines.append("intervention against the flat baseline. Variants differ only in the")
    lines.append("action-type conditioning signal — same Qwen2-VL-2B backbone, same LoRA,")
    lines.append("same data, same hyperparameters, same eval.")
    lines.append("")
    lines.append("- **A (flat baseline)** — plain Qwen2-VL-2B + LoRA. Goal text + image → coordinate.")
    lines.append("- **D (action-conditioned, ours)** — adds an `<|action_slot|>` token to the")
    lines.append("  prompt and `nn.Embedding(8, hidden_dim)` table of trainable action embeddings.")
    lines.append("  At forward time the slot's embedding is replaced by the row for the gold")
    lines.append("  action type.")
    lines.append("")

    for mix in mixes:
        a_key = f"A__{mix}"
        d_key = f"D__{mix}"
        a = summary.get(a_key)
        d = summary.get(d_key)
        if not (a and d):
            continue
        lines.append(f"## data_mix = `{mix}`")
        lines.append("")
        lines.append(f"- n_train: {a.get('n_train')}  ·  epochs: {a.get('epochs')}  ·  seeds: {a.get('n_seeds')}")
        lines.append("")
        lines.append("| Variant | final train loss | hit@0.05 | hit@0.10 | hit@0.25 | mean norm L2 |")
        lines.append("|---|---|---|---|---|---|")
        for v_label, side in (("A (flat baseline)", a), ("D (action-conditioned)", d)):
            lines.append(
                f"| **{v_label}** | "
                f"{_fmt_pm(side['final_loss']['mean'], side['final_loss']['std'], side['n_seeds'])} | "
                f"{_fmt_pm(side['hit_at_005']['mean'], side['hit_at_005']['std'], side['n_seeds'])} | "
                f"**{_fmt_pm(side['hit_at_010']['mean'], side['hit_at_010']['std'], side['n_seeds'])}** | "
                f"{_fmt_pm(side['hit_at_025']['mean'], side['hit_at_025']['std'], side['n_seeds'])} | "
                f"{_fmt_pm(side['mean_norm_l2']['mean'], side['mean_norm_l2']['std'], side['n_seeds'])} |"
            )
        lines.append("")
        # Per-metric delta
        lines.append("Deltas (D − A; positive = D better for hit@r, negative = D better for L2):")
        lines.append("")
        for metric, label, lower_better in (
            ("hit_at_005", "hit@0.05", False),
            ("hit_at_010", "hit@0.10", False),
            ("hit_at_025", "hit@0.25", False),
            ("mean_norm_l2", "mean norm L2", True),
        ):
            dval, pooled, n_sigma = _delta(a, d, metric)
            sign = "+" if dval >= 0 else ""
            verdict = ""
            if n_sigma >= 2.0:
                verdict = " (≥2σ — meaningful)"
            elif n_sigma >= 1.0:
                verdict = " (~1σ)"
            else:
                verdict = " (within noise)"
            lines.append(f"- **{label}**: {sign}{dval:.4f}  ·  pooled std = {pooled:.4f}  ·  {n_sigma:.2f}σ{verdict}")
        lines.append("")

    lines.append("## Interpretation crib sheet (from the project plan)")
    lines.append("")
    lines.append("- **D > A** → decoupling helps. The action embedding carries useful")
    lines.append("  signal that the flat decode can't easily recover from input alone.")
    lines.append("- **D ≈ A** → conditioning doesn't help on this data. Most likely cause:")
    lines.append("  the training distribution doesn't span enough action types for the")
    lines.append("  embedding to learn class-conditional behavior.")
    lines.append("- **D < A** → conditioning is actively harming. Either the embedding is")
    lines.append("  consuming gradient capacity without benefit, or the training schedule")
    lines.append("  isn't long enough to leverage the extra parameters.")
    lines.append("")
    lines.append("Figure: `results/phase4/ablation_A_vs_D.png`")
    lines.append("Raw aggregates: `results/phase4/ablation_summary.json`")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"[p5-render] wrote {OUT}")


if __name__ == "__main__":
    main()

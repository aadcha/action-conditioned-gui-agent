"""Audit Phase 7 result completeness and summarize current metrics.

This intentionally treats Phase 7's low-data and causal-use experiments as
fixed matrices. If any expected cell is missing, the script exits non-zero.

Usage:
    uv run python scripts/p7_result_audit.py
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path


PHASE4 = Path(__file__).resolve().parent.parent / "results" / "phase4"
LOWDATA_NS = [300, 500, 800]
SEEDS = [42, 43, 44]
LOWDATA_FILES = {
    "A": ("variantA", ""),
    "B": ("variantB", "_aux1.0"),
    "Dhook": ("Dhook", "_init0.0"),
}
CAUSAL_PATTERN = (
    "Dtoken_seed{seed}_n1200_ep2_lr2e-05_"
    "mix-all_with_coords_init0.02_causal.json"
)
METRICS = ["hit_at_005", "hit_at_010", "hit_at_025", "mean_normalized_l2"]


def _lowdata_path(variant: str, n_train: int, seed: int) -> Path:
    prefix, suffix = LOWDATA_FILES[variant]
    return PHASE4 / (
        f"{prefix}_seed{seed}_n{n_train}_ep2_lr2e-05_"
        f"mix-all_with_coords{suffix}.json"
    )


def _load_metrics(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text())
    return {metric: data["final_val_metrics"][metric] for metric in METRICS}


def _mean(xs: list[float]) -> float:
    return st.mean(xs) if xs else float("nan")


def _pstdev(xs: list[float]) -> float:
    return st.pstdev(xs) if len(xs) > 1 else 0.0


def main() -> int:
    missing: list[str] = []
    lowdata: dict[str, dict[str, dict[str, object]]] = {}

    for n_train in LOWDATA_NS:
        n_key = str(n_train)
        lowdata[n_key] = {}
        for variant in LOWDATA_FILES:
            rows = []
            for seed in SEEDS:
                path = _lowdata_path(variant, n_train, seed)
                if not path.exists():
                    missing.append(str(path.relative_to(PHASE4.parent.parent)))
                    continue
                rows.append({"seed": seed, **_load_metrics(path)})
            lowdata[n_key][variant] = {
                "n": len(rows),
                "seeds": [row["seed"] for row in rows],
                "means": {metric: _mean([row[metric] for row in rows]) for metric in METRICS},
                "std": {metric: _pstdev([row[metric] for row in rows]) for metric in METRICS},
                "rows": rows,
            }

    causal: dict[str, object] = {"runs": []}
    for seed in SEEDS:
        path = PHASE4 / CAUSAL_PATTERN.format(seed=seed)
        if not path.exists():
            missing.append(str(path.relative_to(PHASE4.parent.parent)))
            continue
        data = json.loads(path.read_text())
        block = data.get("causal_eval")
        if not block:
            missing.append(f"{path.relative_to(PHASE4.parent.parent)} missing causal_eval")
            continue
        causal["runs"].append({
            "seed": seed,
            "action_emb_norm": block["action_emb_norm"],
            "gold": {metric: block["gold"][metric] for metric in METRICS},
            "wrong": {metric: block["wrong"][metric] for metric in METRICS},
            "zero": {metric: block["zero"][metric] for metric in METRICS},
        })

    if missing:
        print("Missing Phase 7 result cells:")
        for item in missing:
            print(f"  - {item}")
        return 1

    summary = {"lowdata": lowdata, "causal": causal}
    out = PHASE4 / "phase7_result_audit.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"Phase 7 audit passed; wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Phase 3.1 — characterize AITW (cjfcsjt/AITW_General, "standard" config).

Streams a sample of rows from the train split, applies our canonical taxonomy,
and dumps the action-type distribution + a handful of per-episode summaries to
results/phase3/aitw_distribution.json. Idempotent.

Usage:
    uv run python scripts/p3_aitw_distribution.py --n 10000

The point of this script is to (a) confirm AITW gives us 5-6 active canonical
classes and (b) provide the dataset summary that the writeup will cite.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.data.aitw import iter_aitw_steps
from src.data.taxonomy import ID_TO_ACTION, CANONICAL_ACTIONS

OUT_PATH = Path(__file__).resolve().parent.parent / "results" / "phase3" / "aitw_distribution.json"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000, help="rows to stream (0 = all)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--config", default="standard")
    args = ap.parse_args()

    print(f"[p3] streaming {args.n} rows from cjfcsjt/AITW_General/{args.config}/{args.split}...")
    steps = list(iter_aitw_steps(
        config=args.config, split=args.split, n_max=args.n, include_images=False
    ))
    print(f"[p3] kept {len(steps)} steps (after dropping unmappable raw ops)")

    raw_int_counts = Counter(s.raw_action_type_int for s in steps)
    string_label_counts = Counter(s.string_label for s in steps)
    canonical_counts = Counter(s.canonical_action_id for s in steps)
    episodes = {s.ep_id for s in steps}
    typed_text_examples = [s.typed_text for s in steps if s.typed_text][:5]
    goal_examples = list({s.goal_info for s in steps})[:5]

    summary = {
        "source": "cjfcsjt/AITW_General",
        "config": args.config,
        "split": args.split,
        "rows_streamed": args.n,
        "rows_kept_after_taxonomy": len(steps),
        "unique_episodes": len(episodes),
        "raw_int_distribution": dict(raw_int_counts),
        "string_label_distribution": dict(string_label_counts),
        "canonical_distribution_count": {
            ID_TO_ACTION[c]: n for c, n in sorted(canonical_counts.items(), key=lambda x: -x[1])
        },
        "canonical_distribution_pct": {
            ID_TO_ACTION[c]: 100.0 * n / len(steps)
            for c, n in sorted(canonical_counts.items(), key=lambda x: -x[1])
        },
        "active_canonical_classes": sorted(
            ID_TO_ACTION[c] for c in canonical_counts
        ),
        "canonical_axes": {k: v for k, v in CANONICAL_ACTIONS.items()},
        "typed_text_examples": typed_text_examples,
        "goal_examples": goal_examples,
    }

    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print(f"[p3] wrote {OUT_PATH}")
    print()
    print("=== AITW canonical distribution ===")
    total = sum(canonical_counts.values())
    for c, n in sorted(canonical_counts.items(), key=lambda x: -x[1]):
        print(f"  {ID_TO_ACTION[c]:<15}  n={n:>5}  ({100 * n / total:5.2f}%)")
    print(f"\n=> {len(canonical_counts)} active classes, "
          f"{sum(1 for n in canonical_counts.values() if n / total >= 0.05)} of which are >=5% of steps")


if __name__ == "__main__":
    main()

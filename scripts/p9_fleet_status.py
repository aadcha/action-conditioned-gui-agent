"""Phase 9 fleet reconciliation: expected run files vs what's on the Volume.

Prints missing runs and the exact relaunch command for each (prespec
failed-run rule: relaunch same seed/config). Usage:

    MODAL_PROFILE=sentinel uv run modal volume ls stage1-cache stage2_runs \
        | uv run python scripts/p9_fleet_status.py
"""

from __future__ import annotations

import sys

FS = "fs-aitw_frozen_v1"
SEEDS = (42, 43, 44)


def expected() -> dict[str, str]:
    out = {}
    base = "--frozen-split aitw_frozen_v1 --eval-test"
    for s in SEEDS:
        awc = f"--n-val 250 --seed {s} --data-mix all_with_coords {base} --n-test 600"
        out[f"variantA_seed{s}_n1200_ep2_lr2e-05_mix-all_with_coords_{FS}.json"] = \
            f"train_stage2_variantA --n-train 1200 {awc}"
        out[f"variantB_seed{s}_n1200_ep2_lr2e-05_mix-all_with_coords_aux1.0_{FS}.json"] = \
            f"train_stage2_variantB --n-train 1200 {awc} --lambda-aux 1.0"
        out[f"variantC_seed{s}_n1200_ep2_lr2e-05_mix-all_with_coords_{FS}.json"] = \
            f"train_stage2_variantC --n-train 1200 {awc}"
        out[f"Dhook_seed{s}_n1200_ep2_lr2e-05_mix-all_with_coords_init0.0_{FS}.json"] = \
            f"train_stage2_Dhook --n-train 1200 {awc} --init-std 0.0"
        out[f"Dtoken_seed{s}_n1200_ep2_lr2e-05_mix-all_with_coords_init0.02_causal_{FS}.json"] = \
            f"train_stage2_Dtoken --n-train 1200 {awc} --init-std 0.02 --causal-eval"
        out[f"e2e_seed{s}_n1200_ep2_lr2e-05_mix-all_with_coords_{FS}.json"] = \
            f"train_stage2_e2e --n-train 1200 {awc}"
        ts = f"--n-val 200 --seed {s} --data-mix taps_and_swipes {base} --n-test 400"
        out[f"variantA_seed{s}_n1000_ep2_lr2e-05_mix-taps_and_swipes_{FS}.json"] = \
            f"train_stage2_variantA --n-train 1000 {ts}"
        out[f"variantB_seed{s}_n1000_ep2_lr2e-05_mix-taps_and_swipes_aux1.0_{FS}.json"] = \
            f"train_stage2_variantB --n-train 1000 {ts} --lambda-aux 1.0"
        out[f"variantC_seed{s}_n1000_ep2_lr2e-05_mix-taps_and_swipes_{FS}.json"] = \
            f"train_stage2_variantC --n-train 1000 {ts}"
        out[f"Dhook_seed{s}_n1000_ep2_lr2e-05_mix-taps_and_swipes_init0.0_{FS}.json"] = \
            f"train_stage2_Dhook --n-train 1000 {ts} --init-std 0.0"
        for n in (300, 500, 800):
            low = f"--n-val 250 --seed {s} --data-mix all_with_coords {base} --n-test 600"
            out[f"variantA_seed{s}_n{n}_ep2_lr2e-05_mix-all_with_coords_{FS}.json"] = \
                f"train_stage2_variantA --n-train {n} {low}"
            out[f"variantB_seed{s}_n{n}_ep2_lr2e-05_mix-all_with_coords_aux1.0_{FS}.json"] = \
                f"train_stage2_variantB --n-train {n} {low} --lambda-aux 1.0"
            out[f"Dhook_seed{s}_n{n}_ep2_lr2e-05_mix-all_with_coords_init0.0_{FS}.json"] = \
                f"train_stage2_Dhook --n-train {n} {low} --init-std 0.0"
    return out


def main() -> None:
    present = set()
    for line in sys.stdin:
        line = line.strip().strip("│ ").split()[0] if line.strip() else ""
        if FS in line:
            present.add(line.split("/")[-1])
    exp = expected()
    missing = [k for k in exp if k not in present]
    print(f"expected {len(exp)}, present {len(exp) - len(missing)}, missing {len(missing)}")
    for k in missing:
        print(f"MISSING {k}")
        print(f"  MODAL_PROFILE=sentinel uv run modal run --detach modal_app.py::{exp[k]}")


if __name__ == "__main__":
    main()

"""Spawn Phase-9 jobs on the DEPLOYED Modal app.

Spawned calls on a deployed app run to completion independently of this
process, so nothing here needs to stay alive (unlike `modal run --detach`,
where killing the local client cancels the remote). Each call id is appended to
results/phase9/launched.jsonl for tracking.

    MODAL_PROFILE=sentinel uv run python scripts/p9_launch.py smoke
    MODAL_PROFILE=sentinel uv run python scripts/p9_launch.py stage1
    MODAL_PROFILE=sentinel uv run python scripts/p9_launch.py full [--only GROUP ...]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal

APP = "action-conditioned-gui-agent"
OUT = Path(__file__).resolve().parent.parent / "results" / "phase9"
SEEDS = [42, 43, 44, 45, 46]
TEN_X = 2e-4   # 10x the shared 2e-5 learning rate, for the action table only


def jobs(group: str) -> list[tuple[str, dict]]:
    """(function, kwargs) pairs. Groups are in priority order."""
    S2 = "_stage2_bigval_remote"
    if group == "smoke":
        common = dict(n_train=8, n_val=24, epochs=1, eval_batch_size=16, check_batch_equivalence=8,
                      out_subdir="bigval_smoke")
        return [("_stage1_bigval_preds_remote", dict(seeds=[42], n_train=8, n_val=24, out_subdir="bigval_smoke"))] + [
            (S2, dict(variant=v, seed=42, e2e=v in ("C", "Dhook", "Dtoken", "Dtext"),
                      action_lr=TEN_X if v == "Dtoken" else 0.0, **common))
            for v in ("A", "B", "Bprompt", "C", "Dhook", "Dtoken", "Dtext")
        ] + [(S2, dict(variant="A", seed=42, drop_type_from_train=True, **common))]
    if group == "stage1":
        return [("_stage1_bigval_preds_remote", dict(seeds=SEEDS))]
    chk = lambda s: 64 if s == 42 else 0
    if group == "core_headline":          # P1: the ranking on the big slice, with end-to-end where it applies
        return [(S2, dict(variant=v, seed=s, e2e=v in ("Dhook", "Dtext"), check_batch_equivalence=chk(s)))
                for v in ("A", "B", "Dhook", "Dtext") for s in SEEDS]
    if group == "core_clean":             # P1: the same four with type events removed from training
        return [(S2, dict(variant=v, seed=s, drop_type_from_train=True, check_batch_equivalence=chk(s)))
                for v in ("A", "B", "Dhook", "Dtext") for s in SEEDS]
    if group == "e2e_more":               # P2: reviewer ask, end-to-end for hard routing and the trained token
        return [(S2, dict(variant="C", seed=s, e2e=True, check_batch_equivalence=chk(s))) for s in SEEDS] + \
               [(S2, dict(variant="Dtoken", seed=s, e2e=True, action_lr=TEN_X, check_batch_equivalence=chk(s))) for s in SEEDS]
    if group == "leak":                   # P3: reviewer ask, aux head that cannot see the answer tokens
        return [(S2, dict(variant="Bprompt", seed=s, check_batch_equivalence=chk(s))) for s in SEEDS]
    if group == "dhook_lr":               # P4: reviewer ask, give the additive table the same 10x the token got
        return [(S2, dict(variant="Dhook", seed=s, action_lr=TEN_X, check_batch_equivalence=chk(s))) for s in SEEDS]
    if group == "control_dtoken":         # P5: reviewer ask, prepended token on the taps-and-swipes control,
        # on the ORIGINAL 200-step slice so it sits in the same table as A/B/C/D-hook/D-text, at the shared LR
        # so it is the same variant as the retracted inputs_embeds run on the fixed input_ids path;
        # decoded one example at a time like the legacy control runs it is compared against
        return [(S2, dict(variant="Dtoken", seed=s, data_mix="taps_and_swipes", n_train=1000, n_val=200,
                          skip_boundary=False, eval_batch_size=1, out_subdir="bigval_control")) for s in (42, 43, 44)]
    raise SystemExit(f"unknown group {group}")


FULL = ["core_headline", "core_clean", "e2e_more", "leak", "dhook_lr", "control_dtoken"]


def main() -> None:
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    groups = FULL if args[0] == "full" else [args[0]]
    if "--only" in args:
        groups = args[args.index("--only") + 1:]
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / "launched.jsonl"
    fns: dict[str, modal.Function] = {}
    n = 0
    for g in groups:
        for fname, kw in jobs(g):
            fn = fns.setdefault(fname, modal.Function.from_name(APP, fname))
            call = fn.spawn(**kw)
            rec = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "group": g, "fn": fname, "kwargs": kw, "call_id": call.object_id}
            with log.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            n += 1
            print(f"spawned {g:15s} {fname:30s} {json.dumps({k: v for k, v in kw.items() if k not in ('out_subdir',)})}")
    print(f"\n{n} calls spawned; ids in {log}")


if __name__ == "__main__":
    main()

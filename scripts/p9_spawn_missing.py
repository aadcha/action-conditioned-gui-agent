"""Spawn missing Phase 9 runs as detached function calls on the DEPLOYED app.

`modal run --detach` clients proved fragile in this environment (client
process death cancelled queued/running calls twice). `.spawn()` against the
deployed app has no local process to keep alive: the call runs entirely
server-side and persists its result to the Volume like every other run.

Usage:
    MODAL_PROFILE=sentinel uv run modal deploy modal_app.py   # once
    MODAL_PROFILE=sentinel uv run python scripts/p9_spawn_missing.py [--dry-run]
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.p9_fleet_status import expected  # noqa: E402

APP = "action-conditioned-gui-agent"
FS = "aitw_frozen_v1"


def volume_files() -> set[str]:
    out = subprocess.run(
        ["uv", "run", "modal", "volume", "ls", "stage1-cache", "stage2_runs"],
        capture_output=True, text=True, check=True).stdout
    files = set()
    for line in out.splitlines():
        m = re.search(r"stage2_runs/(\S+\.json)", line)
        if m:
            files.add(m.group(1))
    return files


def args_for(fname: str):
    """Map an expected run filename to (remote function name, positional args)."""
    m = re.match(
        r"(variantA|variantB|variantC|Dhook|Dtoken)_seed(\d+)_n(\d+)_ep2_lr2e-05"
        r"_mix-(all_with_coords|taps_and_swipes)(_aux1\.0|_init0\.0|_init0\.02_causal)?"
        rf"_fs-{FS}\.json", fname)
    if not m:
        raise ValueError(f"unparseable filename {fname}")
    kind, seed, n_train, mix = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
    awc = mix == "all_with_coords"
    n_val, n_test = (250, 600) if awc else (200, 400)
    base = dict(epochs=2, lr=2e-5, batch_size=1, seed=seed,
                aitw_split="train", coord_scale=1000)
    if kind == "variantA":
        fn = "_stage2_variantA_train_remote"
        args = (n_train, n_val, base["epochs"], base["lr"], base["batch_size"],
                seed, "train", 1000, False, mix, FS, True, n_test)
    elif kind == "variantB":
        fn = "_stage2_variantB_train_remote"
        args = (n_train, n_val, 2, 2e-5, 1, seed, "train", 1000, mix, 1.0,
                FS, True, n_test)
    elif kind == "variantC":
        fn = "_stage2_variantC_train_remote"
        args = (n_train, n_val, 2, 2e-5, 1, seed, "train", 1000, mix,
                FS, True, n_test)
    elif kind == "Dhook":
        fn = "_stage2_variantDhook_train_remote"
        args = (n_train, n_val, 2, 2e-5, 1, seed, "train", 1000, mix, 0.0, 0.0,
                FS, True, n_test)
    elif kind == "Dtoken":
        fn = "_stage2_variantDtoken_train_remote"
        args = (n_train, n_val, 2, 2e-5, 1, seed, "train", 1000, mix, 0.02, True,
                FS, True, n_test)
    else:
        raise ValueError(kind)
    return fn, args


def main() -> None:
    dry = "--dry-run" in sys.argv
    present = volume_files()
    missing = [k for k in expected() if k not in present]
    print(f"present {len(present & set(expected()))}, missing {len(missing)}")
    for fname in missing:
        fn_name, args = args_for(fname)
        print(f"spawn {fn_name}{args}  ->  {fname}")
        if not dry:
            fn = modal.Function.from_name(APP, fn_name)
            call = fn.spawn(*args)
            print(f"  call id: {call.object_id}")


if __name__ == "__main__":
    main()

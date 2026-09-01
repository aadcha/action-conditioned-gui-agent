"""Phase 8 poller: watch the stage1-cache Volume for the artifacts produced by
the Sep 1 2026 pre-submission runs (E1-E6) and report each as it lands.

Files that already existed on the Volume (re-runs of seeds 45/46 and the
taps_and_swipes control) only count once their Created/Modified time is at or
after LAUNCH_AFTER.

Usage:
    uv run python scripts/p8_poll.py --once          # print status, exit 0 iff all done
    uv run python scripts/p8_poll.py --wave 1        # loop until wave 1 is complete
    uv run python scripts/p8_poll.py                 # loop until everything is complete

Emits one stdout line per newly-completed artifact (so it can drive a Monitor),
a line whenever the running-app count changes, "ALL DONE" (exit 0) when the
selected wave is complete, and "STALLED" (exit 1) if no apps are running but
artifacts are still missing.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

LAUNCH_AFTER = datetime(2026, 9, 1, 14, 0)  # PDT, wall clock of the Volume listing
PROFILE = os.environ.get("MODAL_PROFILE", "sentinel")
MIX = "mix-all_with_coords"
LR = "lr2e-05"

WAVE1 = [
    *[f"stage2_runs/variantA_seed{s}_n1200_ep2_{LR}_{MIX}.json" for s in (45, 46)],
    *[f"stage2_runs/variantB_seed{s}_n1200_ep2_{LR}_{MIX}_aux1.0.json" for s in (45, 46)],
    *[f"stage2_runs/variantC_seed{s}_n1200_ep2_{LR}_{MIX}.json" for s in (45, 46)],
    *[f"stage2_runs/Dhook_seed{s}_n1200_ep2_{LR}_{MIX}_init0.0.json" for s in (45, 46)],
    *[f"stage2_runs/Dtoken_seed{s}_n1200_ep2_{LR}_{MIX}_init0.02.json" for s in (45, 46)],
    *[f"stage2_runs/Dtoken_seed{s}_n1200_ep2_{LR}_{MIX}_init0.02_causal.json" for s in (45, 46)],
    *[f"stage2_runs/variantA_seed{s}_n1000_ep2_{LR}_mix-taps_and_swipes.json" for s in (42, 43, 44)],
    *[f"stage2_runs/Dhook_seed{s}_n1000_ep2_{LR}_mix-taps_and_swipes_init0.0.json" for s in (42, 43, 44)],
    "stage2_runs/m2w_A_seed44_n1000_ep2.json",
    "stage2_runs/m2w_Dhook_seed44_n1000_ep2.json",
    "attn_aggregate/Dhook/attn_aggregate_Dhook_seed42_n1200.json",
    "attn_aggregate/Dtoken/attn_aggregate_Dtoken_seed42_n1200.json",
    "qualitative_v2/qualitative_grounding.png",
    "qualitative_v2/render.json",
]
WAVE2 = [
    *[f"stage2_runs/variantA_seed{s}_n{n}_ep2_{LR}_{MIX}.json" for n in (2500, 5000) for s in (43, 44)],
    *[f"stage2_runs/variantB_seed{s}_n{n}_ep2_{LR}_{MIX}_aux1.0.json" for n in (2500, 5000) for s in (43, 44)],
    *[f"stage2_runs/Dhook_seed{s}_n{n}_ep2_{LR}_{MIX}_init0.0.json" for n in (2500, 5000) for s in (43, 44)],
]
WAVES = {1: WAVE1, 2: WAVE2, 0: WAVE1 + WAVE2}


def _modal(*args: str) -> str:
    env = dict(os.environ, MODAL_PROFILE=PROFILE)
    try:
        return subprocess.run(["uv", "run", "modal", *args], capture_output=True, text=True,
                              env=env, timeout=120).stdout
    except Exception as e:  # transient network / CLI failure: treat as empty
        print(f"WARN modal call failed: {e}", flush=True)
        return ""


def list_dir(path: str) -> dict[str, datetime]:
    out = _modal("volume", "ls", "stage1-cache", path, "--json")
    try:
        rows = json.loads(out)
    except Exception:
        return {}
    res = {}
    for r in rows:
        if r.get("Type") != "file":
            continue
        ts = r.get("Created/Modified", "")
        try:
            dt = datetime.strptime(ts.rsplit(" ", 1)[0], "%Y-%m-%d %H:%M")
        except Exception:
            continue
        res[r["Filename"]] = dt
    return res


def running_apps() -> int:
    out = _modal("app", "list", "--json")
    try:
        apps = json.loads(out)
    except Exception:
        return -1
    return sum(1 for a in apps
               if (a.get("Description") or a.get("description")) == "action-conditioned-gui-agent"
               and "detached" in str(a.get("State") or a.get("state")))


def check(expected: list[str]) -> tuple[set[str], set[str]]:
    dirs = sorted({p.rsplit("/", 1)[0] for p in expected})
    seen: dict[str, datetime] = {}
    for d in dirs:
        seen.update(list_dir(d))
    done = {p for p in expected if p in seen and seen[p] >= LAUNCH_AFTER}
    return done, set(expected) - done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", type=int, default=0, choices=(0, 1, 2))
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=int, default=300)
    a = ap.parse_args()
    expected = WAVES[a.wave]
    reported: set[str] = set()
    last_running = None
    while True:
        done, missing = check(expected)
        for p in sorted(done - reported):
            print(f"DONE {p}", flush=True)
        reported |= done
        n_run = running_apps()
        if n_run != last_running:
            print(f"STATUS running_apps={n_run} done={len(done)}/{len(expected)} "
                  f"{time.strftime('%H:%M:%S')}", flush=True)
            last_running = n_run
        if not missing:
            print("ALL DONE", flush=True)
            return 0
        if a.once:
            print("MISSING " + ", ".join(sorted(missing)), flush=True)
            return 1
        if n_run == 0:
            print("STALLED missing=" + ", ".join(sorted(missing)), flush=True)
            return 1
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())

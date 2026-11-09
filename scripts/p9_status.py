"""Status of every call in results/phase9/launched.jsonl: done / running / FAILED (with the error).

    MODAL_PROFILE=sentinel uv run python scripts/p9_status.py [group ...]
"""
import json
import sys
from collections import Counter
from pathlib import Path

import modal

LOG = Path(__file__).resolve().parent.parent / "results" / "phase9" / "launched.jsonl"
groups = set(sys.argv[1:])
rows = [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()]
if groups:
    rows = [r for r in rows if r["group"] in groups]
tally = Counter()
for r in rows:
    kw = r["kwargs"]
    label = f"{r['group']:14s} {kw.get('variant', 'stage1'):8s} seed={kw.get('seed', kw.get('seeds'))}" + \
            (" notype" if kw.get("drop_type_from_train") else "") + (f" alr={kw['action_lr']:g}" if kw.get("action_lr") else "")
    try:
        res = modal.FunctionCall.from_id(r["call_id"]).get(timeout=0)
        tally["done"] += 1
        extra = ""
        if isinstance(res, dict) and "timing" in res:
            t = res["timing"]
            extra = (f"h10={res['hit_at_010']:.3f} train={t.get('train_s', 0)/60:.0f}m eval={t.get('eval_s', 0)/60:.1f}m "
                     f"e2e={t.get('e2e_eval_s', 0)/60:.1f}m total={t['total_s']/60:.0f}m")
            if res.get("batch_check"):
                extra += f" batch_ident={res['batch_check']['frac_identical']:.2f}"
            if res.get("e2e_hit_at_010") is not None:
                extra += f" e2e_h10={res['e2e_hit_at_010']:.3f}"
        print(f"DONE    {label}  {extra}")
    except TimeoutError:
        tally["running"] += 1
        print(f"running {label}")
    except Exception as e:  # remote exception surfaces here
        tally["FAILED"] += 1
        print(f"FAILED  {label}  {type(e).__name__}: {str(e)[:300]}")
print(dict(tally))

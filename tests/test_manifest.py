"""Split-disjointness and integrity tests for the frozen AITW manifest.

These encode the release-gate requirements: no episode or goal crosses splits,
the untouched-test region is respected, and the committed manifest hashes to
its recorded checksum.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

MANIFEST = Path(__file__).resolve().parent.parent / "data" / "manifests" / "aitw_frozen_v1.json"


@pytest.fixture(scope="module")
def manifest():
    if not MANIFEST.exists():
        pytest.skip("manifest not built")
    return json.loads(MANIFEST.read_text())


def test_entries_checksum(manifest):
    canonical = json.dumps(manifest["entries"], sort_keys=True).encode()
    assert hashlib.sha256(canonical).hexdigest() == manifest["entries_sha256"]


def test_episode_disjoint(manifest):
    eps = {}
    for e in manifest["entries"]:
        prev = eps.setdefault(e["ep_id"], e["split"])
        assert prev == e["split"], f"episode {e['ep_id']} crosses splits"


def test_goal_disjoint(manifest):
    goals = {}
    for e in manifest["entries"]:
        prev = goals.setdefault(e["goal_key"], e["split"])
        assert prev == e["split"], f"goal {e['goal_key']!r} crosses splits"


def test_untouched_test_region(manifest):
    lo = manifest["untouched_test_min_awc_index"]
    for e in manifest["entries"]:
        if e["split"] == "test":
            assert e["awc_index"] >= lo, (
                f"test step at awc_index={e['awc_index']} dips into the "
                f"historically-touched region (<{lo})")


def test_counts_match_entries(manifest):
    got = Counter(e["split"] for e in manifest["entries"])
    for split in ("train", "val", "test"):
        assert got[split] == manifest["counts"][split]["steps"]


def test_labels_are_all_with_coords(manifest):
    allowed = set(manifest["filter_labels"])
    assert allowed == {"tap", "swipe_up", "swipe_down", "swipe_left", "swipe_right", "type"}
    assert all(e["label"] in allowed for e in manifest["entries"])


def test_eval_sets_large_enough(manifest):
    """The prespecified eval sets must exist inside the frozen splits."""
    val = [e for e in manifest["entries"] if e["split"] == "val"]
    test = [e for e in manifest["entries"] if e["split"] == "test"]
    ts = {"tap", "swipe_up", "swipe_down", "swipe_left", "swipe_right"}
    assert len(val) >= 250          # awc val eval set
    assert len(test) >= 600         # awc untouched-test eval set
    assert sum(1 for e in val if e["label"] in ts) >= 200    # control val
    assert sum(1 for e in test if e["label"] in ts) >= 400   # control test

"""Pipeline tests — run with:  python tests/test_pipeline.py   (or pytest -q)

These test the three claims the report makes, and each is written to be able to FAIL, so a
green run means something:

  1. perception actually finds objects (not just "runs without error"),
  2. history retrieval actually ranks a same-class prior first,
  3. the reasoner returns the required contract AND does not claim a prior defect the
     history never contained — the failure mode that matters for a decision tool a
     supervisor is meant to trust.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opencv_inspector.detector import Detector            # noqa: E402
from opencv_inspector.line_history import LineHistory      # noqa: E402
from opencv_inspector import reasoning                     # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE = os.path.join(ROOT, "samples", "bus.jpg")


def test_perception_finds_objects():
    import cv2
    img = cv2.imread(SAMPLE)
    assert img is not None, f"sample frame missing at {SAMPLE}"
    dets, ms = Detector().detect(img)
    # the sample is a bus scene with 4 people; requiring >=4 makes a broken model fail
    assert len(dets) >= 4, f"expected >=4 detections, got {len(dets)}"
    assert any(d["class"] == "bus" for d in dets), "the bus in the sample was not detected"
    assert ms < 5000, f"perception took {ms:.0f} ms — implausibly slow for CPU"
    print(f"  perception: {len(dets)} detections in {ms:.0f} ms")


def test_history_ranks_matching_class_first():
    h = LineHistory()
    h.seed([
        {"id": "A1", "station": "S", "class": "truck", "minutes_ago": 5},
        {"id": "A2", "station": "S", "class": "person", "minutes_ago": 60},
        {"id": "A3", "station": "S", "class": "person", "minutes_ago": 10},
    ])
    rows = h.similar("S", "person", limit=3)
    assert rows, "retrieval returned nothing"
    assert rows[0]["class"] == "person", "a different class outranked the matching class"
    # and it must be the more RECENT person first (recency is the second signal)
    assert rows[0]["id"] == "A3", f"expected the recent A3 first, got {rows[0]['id']}"
    print("  history: matching class ranked first and most recent")


def test_reasoner_contract_and_no_invented_history():
    detection = {"class": "surface_crack", "confidence": 0.9, "bbox": [1, 2, 3, 4],
                 "station": "QA-9"}
    history = [{"id": "D-100", "class": "surface_crack", "severity": "low", "minutes_ago": 30}]
    try:
        d = reasoning.reason(detection, history)
    except Exception as exc:
        raise AssertionError(f"reasoner raised (is the LLM endpoint up?): {exc}")

    for key in ("critical", "severity", "matched_prior", "pattern", "action", "reasoning"):
        assert key in d, f"missing contract key: {key}"
    assert isinstance(d["critical"], bool), "critical must be a boolean"
    assert d["severity"] in ("low", "medium", "high"), f"bad severity: {d['severity']}"
    # the invented-history guard: any matched prior must be one we actually passed in
    if d["matched_prior"]:
        assert d["matched_prior"] in {"D-100"}, \
            f"agent cited a prior it was never given: {d['matched_prior']}"
    print(f"  reasoner: severity={d['severity']} critical={d['critical']} "
          f"prior={d['matched_prior']}")


def main():
    failures = []
    for fn in (test_perception_finds_objects, test_history_ranks_matching_class_first,
               test_reasoner_contract_and_no_invented_history):
        print(f"- {fn.__name__}")
        try:
            fn()
        except AssertionError as exc:
            failures.append((fn.__name__, str(exc)))
            print(f"  FAIL: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures.append((fn.__name__, repr(exc)))
            print(f"  ERROR: {exc!r}")
    print()
    if failures:
        print(f"FAILED {len(failures)}/{3}")
        for name, msg in failures:
            print(f"  {name}: {msg}")
        return 1
    print("all 3 checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

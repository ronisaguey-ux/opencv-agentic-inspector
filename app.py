#!/usr/bin/env python3
"""Agentic Visual Inspector — web endpoint.

The competition requires "a working web endpoint or an arranged live screen-share
demonstration". This is that endpoint: upload a plant-floor frame (or use the bundled
sample), the perception stage runs OpenCV 5's DNN detector on CPU, and the reasoning stage
investigates the highest-confidence detection against the line's recent history and returns
a decision with a reasoning trace.

    python app.py            # http://127.0.0.1:5092

Perception is local (OpenCV + ONNX, no cloud). Reasoning is pluggable: the local
OpenAI-compatible backend by default, Amazon Bedrock when REASONING_BACKEND=bedrock.
"""
import base64
import io
import os
import time

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from opencv_inspector.detector import Detector
from opencv_inspector.line_history import LineHistory
from opencv_inspector import reasoning

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

SAMPLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")

_detector = None
_history = LineHistory(os.environ.get("HISTORY_DB", ":memory:"))


def detector():
    global _detector
    if _detector is None:
        _detector = Detector()
    return _detector


def backend():
    if os.environ.get("REASONING_BACKEND", "local") == "bedrock":
        return reasoning.BedrockBackend()
    return reasoning.LocalBackend()


def _seed_history():
    """A believable line: the same class recurring, getting closer together."""
    _history.seed([
        {"id": "D-041", "station": "QA-3", "class": "person", "severity": "low", "minutes_ago": 220},
        {"id": "D-052", "station": "QA-3", "class": "person", "severity": "medium", "minutes_ago": 95},
        {"id": "D-057", "station": "QA-3", "class": "truck", "severity": "low", "minutes_ago": 40},
        {"id": "D-061", "station": "QA-3", "class": "person", "severity": "medium", "minutes_ago": 18},
    ])


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({"ok": True, "service": "opencv-agentic-inspector",
                    "opencv": cv2.__version__, "reasoning_backend": backend().name,
                    "time": time.time()})


@app.route("/api/history")
def api_history():
    station = request.args.get("station", "QA-3")
    return jsonify({"station": station, "recent": _history.recent(station, 20),
                    "stats": _history.stats(station)})


@app.route("/api/inspect", methods=["POST"])
def api_inspect():
    # image: either uploaded, or the bundled sample
    raw = None
    if request.files.get("image"):
        raw = request.files["image"].read()
    else:
        sample = os.path.join(SAMPLES, "bus.jpg")
        if os.path.exists(sample):
            raw = open(sample, "rb").read()
    if not raw:
        return jsonify({"ok": False, "error": "no image and no sample available"}), 400

    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"ok": False, "error": "could not decode image"}), 400

    station = request.form.get("station", "QA-3")
    t0 = time.time()
    dets, ms = detector().detect(img)

    if not dets:
        return jsonify({"ok": True, "detections": [], "latency_ms": round(ms, 1),
                        "decision": None,
                        "note": "no relevant objects detected in this frame"})

    # the inspector investigates the most confident detection
    primary = dets[0]
    primary["station"] = station
    history = _history.similar(station, primary["class"], limit=6)

    decision = None
    try:
        decision = reasoning.reason(primary, history, store=_history, backend=backend())
    except Exception as exc:
        decision = {"error": f"reasoning failed: {exc}"}

    # record it so the line's memory grows (and the next frame has more context)
    defect_id = f"D-{int(time.time()) % 100000}"
    _history.add(defect_id, station, primary["class"],
                 (decision or {}).get("severity", "low"), primary["confidence"])
    decision["defect_id"] = defect_id

    overlay = Detector.render(img, dets)
    ok, buf = cv2.imencode(".jpg", overlay)
    overlay_b64 = base64.b64encode(buf.tobytes()).decode()

    return jsonify({
        "ok": True,
        "detections": dets,
        "primary": primary,
        "history": history,
        "decision": decision,
        "latency_ms": round(ms, 1),
        "total_s": round(time.time() - t0, 1),
        "overlay_jpeg_b64": overlay_b64,
        "image_size": [img.shape[1], img.shape[0]],
    })


if __name__ == "__main__":
    os.makedirs(SAMPLES, exist_ok=True)
    if not _history.recent("QA-3"):
        _seed_history()
    port = int(os.environ.get("PORT", "5092"))
    app.run(host="0.0.0.0", port=port, debug=False)

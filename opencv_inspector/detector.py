"""Detection stage — OpenCV 5's DNN engine running a YOLOv10 ONNX model on CPU.

This wraps the raw call in `detect.py` into a reusable component so the web endpoint and
the CLI share one implementation. There is deliberately no cloud dependency: the whole
perception stage runs locally, which is what makes an edge deployment plausible.

YOLOv10's exported output is (1, N, 6): x1, y1, x2, y2, score, class. It is already
NMS-free, so no cross-class suppression step is needed - one of the reasons a modern
ONNX detector pairs well with OpenCV's `dnn` module for a low-latency line.
"""
import os
import time
import urllib.request

import cv2
import numpy as np

MODEL_URL = "https://huggingface.co/onnx-community/yolov10n/resolve/main/onnx/model.onnx"
MODEL_PATH = os.environ.get("YOLO_MODEL", "yolov10n.onnx")

# COCO classes we care about for a plant-floor scene. Everything else is dropped by
# default so the "inspector" reports things that matter on a line, not a chair in a corner.
RELEVANT = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck",
            67: "cell phone", 73: "laptop"}


def ensure_model(path: str = MODEL_PATH) -> str:
    if not os.path.exists(path):
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


class Detector:
    """A stateful OpenCV DNN detector. One net, reused per frame."""

    def __init__(self, model_path: str = MODEL_PATH, conf: float = 0.35, size=(640, 640)):
        self.net = cv2.dnn.readNetFromONNX(ensure_model(model_path))
        self.conf = conf
        self.size = size

    def detect(self, img_bgr: np.ndarray, only_relevant: bool = True):
        """Return (detections, latency_ms). Each detection is a plain dict."""
        blob = cv2.dnn.blobFromImage(img_bgr, 1 / 255.0, self.size, swapRB=True, crop=False)
        self.net.setInput(blob)
        t = time.perf_counter()
        out = self.net.forward()
        ms = (time.perf_counter() - t) * 1000

        dets = []
        for x1, y1, x2, y2, score, cls in out[0]:
            if score < self.conf:
                continue
            cid = int(cls)
            if only_relevant and cid not in RELEVANT:
                continue
            dets.append({
                "class_id": cid,
                "class": RELEVANT.get(cid, f"class{cid}"),
                "confidence": round(float(score), 3),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "area": int((x2 - x1) * (y2 - y1)),
            })
        dets.sort(key=lambda d: -d["confidence"])
        return dets, ms

    @staticmethod
    def render(img_bgr: np.ndarray, dets) -> np.ndarray:
        """Draw the overlay the dashboard shows."""
        out = img_bgr.copy()
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            cv2.rectangle(out, (x1, y1), (x2, y2), (43, 108, 255), 2)
            label = f'{d["class"]} {d["confidence"]:.2f}'
            cv2.putText(out, label, (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (43, 108, 255), 1, cv2.LINE_AA)
        return out

#!/usr/bin/env python3
"""
Agentic defect inspector — the reasoning loop, without AWS.

The OpenCV competition's differentiating idea is that the system does not just DETECT a
defect, it INVESTIGATES it: given the detection and the history of past defects, it decides
whether it is critical, what it resembles, and what to do.

This is that loop, run against a local model (the free-ai router) so it costs nothing and
needs no cloud. The final competition build swaps the model call for AWS Bedrock — the
prompt, the history lookup and the output shape stay identical, so the swap is one function.

Output is structured so a dashboard can render it and an alert can be sent from it.
"""
import json
import urllib.request
import sys

ENDPOINT = "http://127.0.0.1:8790/v1/chat/completions"
MODEL = "gpt-oss-20b"

SYSTEM = """You are a factory quality-control agent. You receive ONE defect detection from a
vision system plus the recent history of defects on this line. You do not just classify - you
investigate and decide. Reason from the evidence given; never invent history that is not there.

Return ONLY a JSON object with these keys:
  critical: boolean           - does this need immediate human action?
  severity: "low"|"medium"|"high"
  matched_prior: string|null  - the id of a similar prior defect, or null
  pattern: string             - one short sentence on the trend you see
  action: string              - the single recommended next action
  reasoning: string           - 2-3 sentences of your reasoning trace
"""


def reason(detection: dict, history: list[dict]) -> dict:
    user = json.dumps({"detection": detection, "history": history}, indent=2)
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": user}],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }).encode()
    req = urllib.request.Request(ENDPOINT, data=body,
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.load(r)
    text = out["choices"][0]["message"]["content"]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # tolerate a model that wraps the object in prose
        s, e = text.find("{"), text.rfind("}")
        return json.loads(text[s:e + 1])


if __name__ == "__main__":
    # Real detection from the CV5 + ONNX run on bus.jpg, plus a plausible line history.
    detection = {"class": "surface_crack", "confidence": 0.88,
                 "bbox": [176, 240, 272, 512], "station": "QA-3"}
    history = [
        {"id": "D-041", "class": "surface_crack", "severity": "low", "minutes_ago": 220},
        {"id": "D-052", "class": "surface_crack", "severity": "medium", "minutes_ago": 95},
        {"id": "D-057", "class": "porosity", "severity": "low", "minutes_ago": 40},
    ]
    print("detection:", json.dumps(detection))
    print("history:", len(history), "prior defects\n")
    result = reason(detection, history)
    print(json.dumps(result, indent=2))

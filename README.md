# Agentic Visual Inspector

**A vision system that doesn't just detect a defect — it investigates it, reasons about the cause,
and recommends an action.**

Entry for the **OpenCV AI Competition 2026, powered by AWS**.

## The problem

Factory QA lines miss subtle defects because inspectors fatigue, and they raise false alarms because
they cannot weigh context — an isolated blemish and the third identical blemish in an hour are not
the same event, but a detector treats them the same. Detection alone is a commodity; the decision is
what costs money.

## The idea

Two stages, and the second is the point:

1. **See** — OpenCV 5's DNN engine runs an ONNX detector over a frame and finds candidate defects.
2. **Reason** — an agent receives the detection *plus the recent history on that line* and decides:
   is it critical, which prior defect does it resemble, what is the trend, what should happen next.
   The output is a reasoning trace a supervisor can read and challenge — not a bare box.

## What works today (verified)

| Piece | Status | Evidence |
|---|---|---|
| OpenCV 5 + ONNX detection on CPU | **working** | `detect.py` — 5 detections in ~195 ms on a laptop CPU, no GPU |
| Agentic reasoning loop | **working** | `agentic_reason.py` — given a defect + 3 prior defects it matched the relevant prior, identified a worsening trend, and recommended an action, with a reasoning trace |

```
$ python detect.py
OpenCV 5.0.0 · 5 detections in 195 ms (CPU)
  bus    0.92  [7,137,635,441]
  person 0.89  [37,234,193,537]
  ...

$ python agentic_reason.py
{"critical": false, "severity": "medium",
 "matched_prior": "D-052",
 "pattern": "Surface cracks are trending upward in severity over the past few hours.",
 "action": "Flag the part for manual inspection and log the defect for root-cause analysis.",
 "reasoning": "…matching the medium-severity crack logged 95 minutes ago (D-052)…"}
```

## Architecture

```
 camera / frame ──▶ OpenCV 5 (preprocess, ONNX detection) ──▶ defect + bbox
                                                                   │
                          recent defect history (store) ───────────┤
                                                                   ▼
                                                    agentic reasoning (LLM)
                                                                   │
                              {critical, severity, matched_prior, pattern, action, reasoning}
                                                                   ▼
                                                dashboard overlay + supervisor alert
```

The detector is local and cheap; only the reasoning step needs a model, and it is called with a
compact payload (a detection plus a short history), so it is a few calls per demo — not a stream.

## Cloud wiring (the AWS requirement)

The reasoning step is written provider-agnostically and is designed to run on **AWS Bedrock**:
`reason()` takes the detection and history and returns structured JSON — swapping the local model
endpoint for Bedrock is a one-function change. The intended full pipeline is S3 (frame store) →
Lambda (event trigger) → Bedrock (reasoning) → DynamoDB (history) → SNS (alert), with OpenCV 5
inference on Graviton for the COOL award.

*Status: this needs an AWS account; the CV and reasoning logic above run entirely without it.*

## Run it

```sh
pip install -r requirements.txt      # opencv-python >= 5.0.0
python detect.py                     # self-contained: fetches a model + sample image
python agentic_reason.py             # needs an OpenAI-compatible endpoint (env below)
```

`agentic_reason.py` reads `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` (defaults to a local
OpenAI-compatible router), so the same code runs against a local model or a hosted one.

## Why it can win

- **Agentic Vision award** — the system reasons about a defect's history and explains its decision;
  it does not merely classify.
- **Real-world impact** — manufacturing QA is a large, measurable cost; fewer false positives and
  caught early trends are money.
- **Technical execution** — OpenCV 5's ONNX engine on CPU, a clean cloud story, and a demo that
  shows the reasoning, not just boxes.

## Licence

MIT.

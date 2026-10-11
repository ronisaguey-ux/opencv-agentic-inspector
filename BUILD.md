# Build, deploy, test

Verified on: Python 3.12, Linux x86_64, **CPU only** (no GPU, no CUDA).

## Perception (local, required)

```sh
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt        # opencv-python-headless, numpy, flask
python detect.py                       # fetches yolov10n.onnx + bus.jpg on first run
```
`detect.py` prints the detections and writes `detected.jpg`. Expected: **5 detections
(bus + 4 person) in ~110-200 ms on a laptop CPU.**

## Web endpoint (the judge demo)

```sh
python app.py                          # http://127.0.0.1:5092
```
- `GET  /api/health`   -> OpenCV version, reasoning backend
- `GET  /api/history`  -> the line's recent defects
- `POST /api/inspect`  -> multipart `image` (optional; falls back to the bundled sample) and
  `station`. Returns detections + the agent's decision + the annotated frame.

## Reasoning backend

Local (default, zero cost):
```sh
export LLM_BASE_URL=http://127.0.0.1:8790/v1/chat/completions
export LLM_MODEL=gpt-oss-20b
```

AWS / Bedrock (the cloud deployment path):
```sh
pip install boto3
export REASONING_BACKEND=bedrock
export BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0
export AWS_REGION=us-east-1
```
The loop, prompt and output contract are identical in both; only the backend object changes.

## Test

```sh
python -m pytest -q      # or: python tests/test_pipeline.py
```
Covers: the detector finds >=4 objects on the sample; the line-history retrieval ranks a
matching class first; the reasoner returns the required keys and never invents a prior id
that was not in the history it was given.

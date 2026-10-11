# Running this entry

The detector needs **OpenCV 5** (`opencv-python>=5.0.0`). On this machine a working
interpreter already has it, so the entry runs without a fresh install:

```sh
# detection (OpenCV 5 DNN + ONNX on CPU, no cloud)  — verified 126 ms, 5 detections
/home/roni/Roni_workspace/opencv-challenge/.venv/bin/python detect.py

# agentic reasoning loop (needs an OpenAI-compatible endpoint; free-ai router on :8790)
python3 agentic_reason.py
```

For a clean machine: `pip install -r requirements.txt` then the same two commands
with `python`. Verified on this box: `cv2 5.0.0`, `numpy 2.5.3`.

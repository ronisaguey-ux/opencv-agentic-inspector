"""Reasoning stage — the agentic investigator.

Detection is a commodity; the decision is what costs a plant money. This stage takes a
detection plus the recent history of the line and produces a judgement a supervisor can
read and challenge: is it critical, does it resemble a prior defect, what is the trend,
what should happen next, and the reasoning trace behind it.

The agent is a bounded tool loop, not a single prompt:

  1. It receives the detection and the retrieved prior defects.
  2. It MAY call `get_line_history(station, limit)` to widen the window if the default
     retrieval looks thin - up to MAX_TOOL_CALLS times, so it can never spin.
  3. It returns a structured decision.

The model backend is pluggable. Locally it is any OpenAI-compatible endpoint (the free-ai
router on :8790). On AWS the identical call is Amazon Bedrock's Converse API - same prompt,
same output shape - so `BedrockBackend` below is a drop-in and the loop does not change.
That is the "meaningful AWS component": the agent's inference is the cloud half, while
perception stays on the edge.

Nothing here invents history. Every fact the agent reasons over is passed in or fetched
from the store; the system prompt forbids claiming history that is not present.
"""
import json
import os
import urllib.request

MAX_TOOL_CALLS = 2

LOCAL_ENDPOINT = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8790/v1/chat/completions")
LOCAL_MODEL = os.environ.get("LLM_MODEL", "gpt-oss-20b")

SYSTEM = """You are a factory quality-control agent. You receive ONE defect detection from a
vision system plus the recent history of defects on this line. You do not merely classify -
you investigate and decide. Reason only from the evidence given; never invent history that is
not there. If the history provided is too thin to judge a trend, you may call the tool
get_line_history to fetch more.

Return ONLY a JSON object with these keys:
  critical: boolean           - does this need immediate human action?
  severity: "low"|"medium"|"high"
  matched_prior: string|null  - the id of the most similar prior defect, or null
  pattern: string             - one short sentence on the trend you see
  action: string              - the single recommended next action
  reasoning: string           - 2-3 sentences of your reasoning trace
"""

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_line_history",
        "description": "Fetch more prior defects for a station when the provided history is too thin.",
        "parameters": {
            "type": "object",
            "properties": {
                "station": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "required": ["station"],
        },
    },
}]


class LocalBackend:
    """Any OpenAI-compatible endpoint. Zero cost; the default for local development."""

    name = "local-openai-compatible"

    def __init__(self, endpoint=LOCAL_ENDPOINT, model=LOCAL_MODEL):
        self.endpoint = endpoint
        self.model = model

    def chat(self, messages, tools=None):
        body = {"model": self.model, "messages": messages, "temperature": 0.2,
                "max_tokens": 2000}
        if tools:
            body["tools"] = tools
        req = urllib.request.Request(self.endpoint, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            out = json.load(r)
        return out["choices"][0]["message"]


class BedrockBackend:
    """Amazon Bedrock Converse API. The AWS deployment path.

    Kept here as the swap point: the loop, the prompt and the output contract are identical
    to LocalBackend, so moving inference to Bedrock changes one line of configuration, not
    the system. Import of boto3 is lazy so the local build needs no AWS SDK.
    """

    name = "aws-bedrock"

    def __init__(self, model_id=None, region=None):
        import boto3  # lazy: only needed on the AWS path
        self.model_id = model_id or os.environ.get(
            "BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0")
        self.client = boto3.client("bedrock-runtime",
                                   region_name=region or os.environ.get("AWS_REGION", "us-east-1"))

    def chat(self, messages, tools=None):
        # Map the OpenAI-style messages onto Converse's shape.
        system = [{"text": m["content"]} for m in messages if m["role"] == "system"]
        conv = [{"role": m["role"], "content": [{"text": m.get("content") or ""}]}
                for m in messages if m["role"] in ("user", "assistant")]
        kw = {}
        if tools:
            kw["toolConfig"] = {"tools": [{"toolSpec": {
                "name": t["function"]["name"],
                "description": t["function"]["description"],
                "inputSchema": {"json": t["function"]["parameters"]},
            }} for t in tools]}
        resp = self.client.converse(modelId=self.model_id, system=system,
                                    messages=conv, inferenceConfig={"temperature": 0.2}, **kw)
        msg = resp["output"]["message"]
        text = "".join(b.get("text", "") for b in msg["content"])
        tool_calls = [b["toolUse"] for b in msg["content"] if "toolUse" in b]
        return {"content": text, "tool_calls": tool_calls}


def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        s, e = (text or "").find("{"), (text or "").rfind("}")
        if s >= 0 and e > s:
            return json.loads(text[s:e + 1])
        raise


def reason(detection: dict, history: list, store=None, backend=None) -> dict:
    """Investigate one detection. Returns a decision dict plus a trace of the steps taken.

    `store` is optional: a callable(station, limit) -> list used by the get_line_history tool.
    """
    backend = backend or LocalBackend()
    trace = []

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps({"detection": detection, "history": history},
                                               indent=2)},
    ]
    trace.append({"step": "perceive", "detail": f"{len(history)} prior defects retrieved"})

    for _ in range(MAX_TOOL_CALLS + 1):
        msg = backend.chat(messages, tools=TOOLS if store else None)
        calls = msg.get("tool_calls") or []
        if not calls:
            break
        messages.append({"role": "assistant", "content": msg.get("content") or ""})
        for call in calls:
            fn = (call.get("function") or {})
            name = fn.get("name") or call.get("name")
            args = fn.get("arguments") or call.get("input") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if name == "get_line_history" and store:
                station = args.get("station") or detection.get("station", "")
                limit = int(args.get("limit") or 20)
                more = store(station, limit)
                trace.append({"step": "tool:get_line_history",
                              "detail": f"{station} -> {len(more)} defects"})
                messages.append({"role": "tool", "content": json.dumps(more)})
            else:
                messages.append({"role": "tool", "content": "[]"})

    decision = _parse_json(msg.get("content") or "")
    decision["trace"] = trace
    decision["backend"] = backend.name
    return decision

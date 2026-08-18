"""
Send ONE test generation span to Langfuse and confirm the v4 wiring works.

Run from backend-python with your real .env in place:
    python scripts/verify_langfuse.py

It reuses the app's own observability.py, so a success here means the running
app will trace correctly too. Then open Langfuse -> Tracing and look for a
trace named "llm-generation" (type GENERATION) with input/output populated.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import os
import observability as obs

host = os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL") or "https://cloud.langfuse.com"
if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
    sys.exit("LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set — nothing to send.")

print(f"Target project host: {host}")
obs.init_observability()          # prints "LLM tracing enabled via OTLP: ..." if keys are good

with obs.llm_generation("verify-script/test-model", "ping: is Langfuse v4 receiving traces?") as span:
    obs.set_generation_output(span, "pong: if you can read this in the Langfuse UI, v4 ingestion works.")

obs.flush()                       # force the BatchSpanProcessor to send now
print("Sent 1 generation span. Open Langfuse -> Tracing and look for 'llm-generation'.")

#!/usr/bin/env python3
"""
BLOOM Multimodal Bridge
========================
Routes BLOOM's visual features (age, gaze, gesture, body_language, spinning) 
to the multimodal inference server for prediction.

Run alongside bloom_multimodal_trainer.py --serve
"""
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Optional


def resolve_server_url(default_url: str) -> str:
    runtime = Path("bloom_multimodal_runtime.json")
    if not runtime.exists():
        return default_url
    try:
        data = json.loads(runtime.read_text(encoding="utf-8"))
        port = int(data.get("port", 8766))
        return f"ws://localhost:{port}"
    except Exception:
        return default_url

async def call_multimodal_server(
    age: Optional[int] = None,
    eye_gaze: str = "unknown",
    face_gesture: str = "unknown",
    body_language: str = "unknown",
    spinning_type: str = "none",
    server_url: str = "ws://localhost:8766"
) -> Dict:
    """Call multimodal inference server and get response."""
    try:
        import websockets
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "websockets", "-q"])
        import websockets
    
    req = {
        "type": "predict",
        "age": age or 8,
        "eye_gaze": eye_gaze,
        "face_gesture": face_gesture,
        "body_language": body_language,
        "spinning_type": spinning_type,
    }

    retry = 3
    last_err = None
    server_url = resolve_server_url(server_url)
    for attempt in range(1, retry + 1):
        try:
            async with websockets.connect(server_url, max_size=10*1024*1024, ping_interval=30, ping_timeout=30) as ws:
                await ws.send(json.dumps(req))
                resp = await ws.recv()
                return json.loads(resp)
        except Exception as e:
            last_err = e
            print(f"  Bridge retry {attempt}/{retry} failed: {e}", file=sys.stderr)
            if attempt < retry:
                await asyncio.sleep(2)

    return {
        "type": "error",
        "message": str(last_err) if last_err else "multimodal server unreachable",
        "fallback_response": "That sounds really interesting! Tell me more — I want to hear everything you want to share.",
    }

async def bridge_example():
    """Example usage."""
    print("🌸 BLOOM Multimodal Bridge Example\n")
    
    # Example 1: Calm child
    print("Test 1: 7-year-old, calm, centered gaze")
    result = await call_multimodal_server(
        age=7,
        eye_gaze="center",
        face_gesture="happy",
        body_language="calm",
        spinning_type="none",
    )
    print(f"  Response: {result.get('text', result.get('message'))}\n")
    
    # Example 2: Stimming child
    print("Test 2: 6-year-old, hand flapping, agitated")
    result = await call_multimodal_server(
        age=6,
        eye_gaze="scattered",
        face_gesture="focused",
        body_language="active",
        spinning_type="none",
    )
    print(f"  Response: {result.get('text', result.get('message'))}\n")
    
    # Example 3: Upset child
    print("Test 3: 10-year-old, upset, body spinning")
    result = await call_multimodal_server(
        age=10,
        eye_gaze="down",
        face_gesture="uncertain",
        body_language="spinning",
        spinning_type="spinning",
    )
    print(f"  Response: {result.get('text', result.get('message'))}\n")

if __name__ == "__main__":
    asyncio.run(bridge_example())

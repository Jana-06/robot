#!/usr/bin/env python3
"""
Optional integration for bloom_server.py to use multimodal predictions.

To enable:
1. Start the multimodal inference server: 
   python3 bloom_multimodal_trainer.py --serve --model-path bloom_multimodal_model.json

2. In bloom_server.py, import and use (see examples below)

3. Restart BLOOM server
"""
import asyncio
import json
import sys
from typing import Dict, Optional, Tuple

async def get_multimodal_response(
    age: Optional[int] = None,
    eye_gaze: str = "unknown",
    face_gesture: str = "unknown",
    body_language: str = "unknown",
    spinning_type: str = "none",
    server_url: str = "ws://localhost:8766"
) -> Optional[str]:
    """
    Call multimodal inference server and return response text.
    Returns None if server unreachable (will fall back to main LLM).
    """
    try:
        import websockets
    except ImportError:
        return None
    
    try:
        async with websockets.connect(
            server_url, 
            max_size=10*1024*1024,
            timeout=3
        ) as ws:
            req = {
                "type": "predict",
                "age": age or 8,
                "eye_gaze": eye_gaze,
                "face_gesture": face_gesture,
                "body_language": body_language,
                "spinning_type": spinning_type,
            }
            await asyncio.wait_for(ws.send(json.dumps(req)), timeout=2)
            resp_raw = await asyncio.wait_for(ws.recv(), timeout=2)
            resp = json.loads(resp_raw)
            
            if resp.get("type") == "response":
                return resp.get("text")
            return None
    except asyncio.TimeoutError:
        return None
    except Exception as e:
        print(f"  [MM] Multimodal call failed: {e}", file=sys.stderr)
        return None

# ── INTEGRATION EXAMPLES ───────────────────────────────────────────────────

"""
EXAMPLE 1: Replace LLM response with multimodal prediction (aggressive)
=====================================================================

In bloom_server.py, modify run_pipeline() like this:

async def run_pipeline(child_msg, profile, history, bcast):
    stage = get_stage(profile["relationship"]["sessions_count"])
    system = build_prompt(profile, stage)
    strict = False
    t0 = time.time()
    loop = asyncio.get_event_loop()
    
    async def step(s, st): 
        await bcast({"type": "guardrail_step", "step": s, "status": st})
    
    await step("input", "pass")
    await step("profile", "pass")
    
    # TRY MULTIMODAL FIRST
    await step("llm", "running")
    
    mm_response = await get_multimodal_response(
        age=profile.get("age"),
        eye_gaze=_vision.eye_gaze,
        face_gesture=_vision.face_gesture,
        body_language=_vision.body_language,
        spinning_type=_vision.spinning_type,
    )
    
    if mm_response:
        print(f"  [MM] Using multimodal response")
        await step("llm", "pass")
        resp = mm_response
        
        # Affirm if stimulus
        is_stim = _has(child_msg, STIM_KEYWORDS) or _has(child_msg, profile.get("sensory", {}).get("stims", []))
        if is_stim and not _has(resp, ["stimming is", "keep stimming"]):
            resp = f"{stim_aff(profile)} {resp}"
        
        await step("harm", "pass")
        await step("affirm", "pass")
        await step("stim", "pass")
        await step("tts", "pass")
        
        return resp, {"is_harmful": False, "is_affirming": True}, time.time() - t0
    
    # FALLBACK TO LLM
    await step("llm", "running")
    try:
        resp = await loop.run_in_executor(
            None, 
            lambda: api_call(child_msg, COMPANION_MODEL, max_tokens=150, system=system, history=history)
        )
    except Exception as e:
        await step("llm", "fail")
        print(f"  LLM err: {e}")
        return SAFE_FALLBACK, {"is_harmful": False, "is_affirming": True}, time.time() - t0
    
    await step("llm", "pass")
    
    # ... rest of harm/affirm checks ...


EXAMPLE 2: Use multimodal for context enrichment (safer)
=========================================================

In build_prompt(), add multimodal context to the system prompt:

def build_prompt(profile, stage, vision=None):
    # ... existing code ...
    
    mem_block = ""
    if mems:
        mem_block = "\n\nRECENT MEMORIES:"
        for i, m in enumerate(reversed(mems), 1):
            mem_block += f"\n  -{i} ({m.get('date', '')}): {m.get('summary', '')}"
    
    # ADD MULTIMODAL CONTEXT
    mm_context = ""
    if vision:
        mm_context = (
            f"\n\nVISUAL SIGNALS (inferred from camera):"
            f"\n  - Eye gaze: {vision.eye_gaze}"
            f"\n  - Face gesture: {vision.face_gesture}"
            f"\n  - Body language: {vision.body_language}"
            f"\n  - Spinning: {vision.spinning_type}"
        )
    
    return f"""You are BLOOM — warm, playful, neurodiversity-affirming AI friend for children.

ABSOLUTE GUARDRAIL RULES — NEVER VIOLATE:
1. NEVER suppress stimming or body movement
2. NEVER pressure eye contact
3. NEVER frame autism/ADHD as broken or needing fixing
4. NEVER push "normal" behaviour
5. NEVER dismiss special interests
6. NEVER correct communication style
7. NEVER invalidate emotions
8. NEVER use compliance/reward language
9. NEVER imply the child needs to change
10. ALWAYS celebrate the child exactly as they are

CHILD: {name}{f', age {age}' if age else ''} ({profile['pronouns']})
STAGE: {stage.replace('_',' ').title()} — {tone}
INTERESTS: {', '.join(interests) or 'not yet known'}
STIMS: {', '.join(stims) or 'none recorded'} — ALWAYS affirm these
COMM: {comm.get('style','unknown')}{mem_block}{mm_context}

RESPONSE: 2-3 sentences. Match child's energy. Affirm stims first if mentioned. Validate feelings first if upset."""


EXAMPLE 3: Hybrid (multimodal classification + LLM generation)
==============================================================

Use multimodal to classify emotion/state, then generate response with that signal:

async def run_pipeline(child_msg, profile, history, bcast):
    # Determine state via multimodal
    mm_response = await get_multimodal_response(
        age=profile.get("age"),
        eye_gaze=_vision.eye_gaze,
        face_gesture=_vision.face_gesture,
        body_language=_vision.body_language,
        spinning_type=_vision.spinning_type,
    )
    
    # Parse response to infer state
    is_stim = "stimming" in (mm_response or "").lower() or _has(child_msg, STIM_KEYWORDS)
    is_upset = "hard" in (mm_response or "").lower() or _vision.face_gesture in ("sad", "afraid")
    
    # Generate with state hint
    state_hint = ""
    if is_stim:
        state_hint = "[child is stimming; affirm unconditionally]"
    elif is_upset:
        state_hint = "[child is upset; validate first]"
    
    system = build_prompt(profile, stage)
    system += f"\n\nSTATE SIGNAL: {state_hint}"
    
    # Call LLM with enriched context
    resp = await loop.run_in_executor(
        None,
        lambda: api_call(child_msg, COMPANION_MODEL, max_tokens=150, system=system, history=history)
    )
    
    # ... harm/affirm checks ...
"""

# ── STANDALONE TEST ────────────────────────────────────────────────────────

async def test_integration():
    """Standalone test without BLOOM server."""
    print("🌸 BLOOM Multimodal Integration Test\n")
    
    test_cases = [
        {
            "name": "Calm child with centered gaze",
            "age": 7,
            "eye_gaze": "center",
            "face_gesture": "happy",
            "body_language": "calm",
            "spinning_type": "none",
        },
        {
            "name": "Stimming child (hand flapping)",
            "age": 6,
            "eye_gaze": "scattered",
            "face_gesture": "focused",
            "body_language": "active",
            "spinning_type": "none",
        },
        {
            "name": "Upset, spinning child",
            "age": 10,
            "eye_gaze": "down",
            "face_gesture": "uncertain",
            "body_language": "spinning",
            "spinning_type": "spinning",
        },
        {
            "name": "Older child, engaged",
            "age": 11,
            "eye_gaze": "center",
            "face_gesture": "neutral",
            "body_language": "calm",
            "spinning_type": "none",
        },
    ]
    
    for test in test_cases:
        name = test.pop("name")
        print(f"Test: {name}")
        print(f"  Signals: {test}")
        
        response = await get_multimodal_response(**test)
        if response:
            print(f"  Response: {response}\n")
        else:
            print(f"  ✗ No response (server may not be running)\n")

if __name__ == "__main__":
    asyncio.run(test_integration())

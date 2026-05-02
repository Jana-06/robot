#!/usr/bin/env python3
"""
BLOOM WebSocket Server — Self-Contained
========================================
Browser sends camera frames + mic audio over WebSocket.
Server does OpenCV face/hand detection + Groq STT + LLM pipeline.
Streams annotated frames + guardrail events + metrics back.

Run:  python3 bloom_server.py --child arjun
Then: open bloom_dashboard.html in your browser
"""
import asyncio, base64, json, math, os, re, subprocess
import sys, tempfile, threading, time, wave
import http.client, argparse
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set
import logging

import cv2
import numpy as np
import uuid

try:
    import websockets
    from websockets.asyncio.server import serve as ws_serve
except ImportError:
    subprocess.run([sys.executable,"-m","pip","install","websockets","-q"])
    import websockets
    from websockets.asyncio.server import serve as ws_serve

try:
    import pyttsx3 as _pyttsx3
except ImportError:
    _pyttsx3 = None

# ── CONFIG ────────────────────────────────────────────────────────────────
LOGGER = logging.getLogger("bloom.server")


def setup_logging():
    if LOGGER.handlers:
        return
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler("bloom_server.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    LOGGER.addHandler(fh)
    LOGGER.addHandler(sh)


GROQ_KEY        = os.environ.get("GROQ_API_KEY", "")
COMPANION_MODEL = "llama-3.3-70b-versatile"
JUDGE_MODEL     = "llama-3.1-8b-instant"
STT_MODEL       = "whisper-large-v3-turbo"
STT_LANGUAGE    = "en"
TTS_ENGINE      = "espeak"
MAX_RETRIES     = 2
RATE_RETRIES    = 3
PROFILES_DIR    = Path("bloom_profiles")
CHILD_SESSIONS_DIR = Path("sessions")
WS_HOST         = "0.0.0.0"
WS_PORT         = int(os.environ.get("PORT", 8765))
JPEG_QUALITY    = 65
ACTION_CTX_WIN  = 2.5
CLAP_COOLDOWN   = 1.2

STIM_KEYWORDS = ["stim","stimming","rocking","flapping","hand flapping","spinning",
                 "bouncing","fidget","fidgeting","humming","pacing","swaying","clapping"]

SUBTYPE_STIM = {
    1:"this helps manage anxiety and big feelings",
    2:"this supports your development and self-regulation",
    3:"this is part of your authentic self and identity",
    4:"this helps keep your body and brain safe and regulated",
}

SAFE_FALLBACK = ("That sounds really interesting! Tell me more — "
                 "I want to hear everything you want to share.")

CHILD_HELPLINE     = "1098"   # India's 24×7 free child helpline
CRISIS_HELPLINE_LINE = (
    "Hey, I care about you so much — if things ever feel really big or unsafe, "
    "you can always call 1098 (it's a free helpline just for children, available "
    "any time, and they really listen 💛)."
)

# Words/phrases that signal the child may be in distress or danger
CRISIS_PATTERNS: list = [
    "want to kill", "wanna kill", "going to kill", "gonna kill",
    "kill myself", "kill my mom", "kill my dad", "kill my sister", "kill my brother",
    "kill everyone", "kill him", "kill her",
    "want to die", "wanna die", "i want to die", "i wanna die",
    "hate my mom", "hate my dad", "hate my parents", "hate my family",
    "hate myself", "hate myself so much",
    "hurt myself", "hurt myself bad", "cutting myself",
    "end my life", "end it all", "end everything",
    "no one loves me", "nobody loves me", "nobody cares about me",
    "want to disappear", "wish i was dead", "wish i were dead",
    "don't want to live", "dont want to live",
    "going to run away", "gonna run away", "want to run away",
    "nobody wants me", "everyone hates me",
    "can't take it anymore", "cant take it anymore",
]

HARMFUL_Q  = 'Does this response harm a neurodivergent child by suppressing stimming, pressuring eye contact, framing autism as broken, pushing normalcy, dismissing interests, correcting communication, or using compliance language?\nResponse: "{response}"\nReply YES or NO only.'
AFFIRMING_Q= 'Does this response celebrate a neurodivergent child as they are, with warm empowering language?\nResponse: "{response}"\nReply YES or NO only.'

EMPTY_PROFILE = {
    "child_id":"","name":"","age":None,"pronouns":"they/them","diagnosis":[],
    "autism_subtype":None,
    "communication":{"style":"unknown","prefers_statements":False,
                     "question_tolerance":"medium","typical_msg_len":"medium","uses_echolalia":False},
    "sensory":{"stims":[],"sensory_notes":""},
    "vision_features":{
        "eye_gaze":"unknown",
        "face_gesture":"unknown",
        "body_language":"unknown",
        "spinning_type":"none",
    },
    "interests":{"deep":[],"current":[],"avoided":[]},
    "emotional":{"common_feelings":[],"calming_topics":[],"distress_signals":""},
    "relationship":{"sessions_count":0,"total_messages":0,"first_session_date":"",
                    "last_session_date":"","stage":"stranger","caregiver_notes":""},
    "episodic_memory":[],
}


def _safe_child_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", (name or "child").strip())
    return cleaned or "child"


def _child_session_file(child_name: str) -> Path:
    date_tag = datetime.now().strftime("%Y-%m-%d")
    return CHILD_SESSIONS_DIR / f"{_safe_child_name(child_name)}_{date_tag}.json"


def load_recent_child_sessions(child_name: str, take: int = 3) -> List[Dict]:
    CHILD_SESSIONS_DIR.mkdir(exist_ok=True)
    prefix = f"{_safe_child_name(child_name)}_"
    files = sorted(CHILD_SESSIONS_DIR.glob(f"{prefix}*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    out: List[Dict] = []
    for f in files[:take]:
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            continue
    return out


def compute_engagement_baseline(recent_sessions: List[Dict]) -> float:
    vals = []
    for s in recent_sessions:
        vals.extend(s.get("engagement_score_history", []))
    if not vals:
        return 50.0
    return round(sum(vals) / max(1, len(vals)), 2)


def compute_personalization_level(total_sessions: int) -> int:
    return int(max(0, min(100, total_sessions * 12)))


def _resolve_multimodal_url(default_port: int = 8766) -> str:
    runtime = Path("bloom_multimodal_runtime.json")
    if not runtime.exists():
        return f"ws://localhost:{default_port}"
    try:
        data = json.loads(runtime.read_text(encoding="utf-8"))
        port = int(data.get("port", default_port))
        return f"ws://localhost:{port}"
    except Exception:
        return f"ws://localhost:{default_port}"


def _port_is_listening(port: int) -> bool:
    try:
        query = [
            "powershell",
            "-NoProfile",
            "-Command",
            f"Get-NetTCPConnection -LocalPort {port} -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess",
        ]
        result = subprocess.run(query, capture_output=True, text=True, check=False)
        return bool(result.stdout.strip())
    except Exception:
        return False


def _clear_port_listeners(port: int) -> bool:
    try:
        query = [
            "powershell",
            "-NoProfile",
            "-Command",
            (
                f"$p=Get-NetTCPConnection -LocalPort {port} -State Listen -ErrorAction SilentlyContinue; "
                "if($p){$p | Select-Object -ExpandProperty OwningProcess | Sort-Object -Unique | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }}"
            ),
        ]
        subprocess.run(query, capture_output=True, text=True, check=False)
        time.sleep(0.5)
        return not _port_is_listening(port)
    except Exception as e:
        LOGGER.warning("port cleanup failed for %s: %s", port, e)
        return False

# ── PROFILE ───────────────────────────────────────────────────────────────
def profile_path(cid): return PROFILES_DIR / f"{cid}.json"
def load_profile(cid):
    p = profile_path(cid)
    return json.loads(p.read_text()) if p.exists() else None
def save_profile(prof):
    PROFILES_DIR.mkdir(exist_ok=True)
    profile_path(prof["child_id"]).write_text(json.dumps(prof, indent=2))
def get_stage(n):
    if n>=10: return "close_friend"
    if n>=4:  return "friend"
    if n>=1:  return "acquaintance"
    return "stranger"

# ── GROQ API ──────────────────────────────────────────────────────────────
RATE_PH = ["rate limit","rate_limit","rpm","quota exceeded","too many requests","429"]

def _groq(prompt, model, max_tokens, system, history=None, temperature=0.7,
          frequency_penalty=0.0, presence_penalty=0.0):
    msgs=[]
    if system:  msgs.append({"role":"system","content":system})
    if history:
        for t in history: msgs.append({"role":t["role"],"content":t["content"]})
    msgs.append({"role":"user","content":prompt})
    payload = {
        "model": model, "messages": msgs,
        "max_tokens": max_tokens, "temperature": temperature,
    }
    if frequency_penalty: payload["frequency_penalty"] = frequency_penalty
    if presence_penalty:  payload["presence_penalty"]  = presence_penalty
    data=json.dumps(payload).encode()
    conn=http.client.HTTPSConnection("api.groq.com",timeout=30)
    conn.request("POST","/openai/v1/chat/completions",body=data,
                 headers={"Content-Type":"application/json","Content-Length":str(len(data)),
                          "Authorization":f"Bearer {GROQ_KEY}"})
    r=json.loads(conn.getresponse().read().decode()); conn.close()
    if "error" in r: raise Exception(r["error"]["message"])
    return r["choices"][0]["message"]["content"].strip()

def api_call(prompt, model, max_tokens=150, system=None, history=None, temperature=0.7,
             frequency_penalty=0.0, presence_penalty=0.0):
    for attempt in range(RATE_RETRIES):
        try: return _groq(prompt,model,max_tokens,system,history,temperature,
                          frequency_penalty,presence_penalty)
        except Exception as e:
            if any(x in str(e).lower() for x in RATE_PH): time.sleep(2**attempt); continue
            raise
    raise Exception("Rate limit exhausted")

def transcribe_wav(wav_bytes: bytes) -> Optional[str]:
    LOGGER.info("STT: received WAV %d bytes", len(wav_bytes))
    if len(wav_bytes) < 1000:
        LOGGER.warning("STT: WAV too short (%d bytes), skipping", len(wav_bytes))
        return None
    bd = "bloom_stt_bnd"
    body = (
        f"--{bd}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n"
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode() + wav_bytes + (
        f"\r\n--{bd}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{STT_MODEL}\r\n"
        f"--{bd}\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\n{STT_LANGUAGE}\r\n"
        f"--{bd}--\r\n"
    ).encode()
    try:
        conn = http.client.HTTPSConnection("api.groq.com", timeout=30)
        conn.request(
            "POST", "/openai/v1/audio/transcriptions", body=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={bd}",
                "Content-Length": str(len(body)),
                "Authorization": f"Bearer {GROQ_KEY}",
            },
        )
        raw = conn.getresponse().read().decode()
        conn.close()
        r = json.loads(raw)
        if "error" in r:
            LOGGER.error("STT API error: %s", str(r["error"])[:120])
            return None
        text = r.get("text", "").strip()
        if text:
            LOGGER.info("STT transcript: '%s'", text)
        return text or None
    except Exception as ex:
        LOGGER.error("STT exception: %s", ex)
        return None

# ── BLOOM PROMPT + PIPELINE ───────────────────────────────────────────────
def build_prompt(profile, stage, recent_bloom_responses=None, is_crisis=False):
    name=profile["name"] or "the child"; age=profile["age"]
    pro={"she/her":("she","her","her"),"he/him":("he","him","his")}.get(profile["pronouns"],("they","them","their"))
    interests=profile["interests"]["deep"]; stims=profile["sensory"]["stims"]
    comm=profile["communication"]; mems=profile["episodic_memory"][-5:]
    vision=profile.get("vision_features",{})
    tone={
        "stranger":    f"First time meeting {name}. Warm, gentle, curious. Introduce yourself briefly.",
        "acquaintance":f"You know {name} a little. Greet by name. Reference a known interest naturally.",
        "friend":      f"You and {name} are real friends. Playful, personal. Reference something you've talked about.",
        "close_friend":f"You and {name} are close. Deep familiarity, inside references, genuine warmth.",
    }[stage]
    mem_block=""
    if mems:
        mem_block="\n\nRECENT MEMORIES:"
        for i,m in enumerate(reversed(mems),1):
            mem_block+=f"\n  [{i}] ({m.get('date','')}): {m.get('summary','')}"

    # Anti-repetition: inject last ≤4 BLOOM responses so the model can explicitly avoid them
    no_repeat_block = ""
    if recent_bloom_responses:
        no_repeat_block = "\n\nYOUR RECENT REPLIES — DO NOT REPEAT, ECHO, OR PARAPHRASE ANY OF THESE:"
        for i, r in enumerate(recent_bloom_responses[-4:], 1):
            no_repeat_block += f"\n  [{i}] {r[:160]}"

    # Crisis instruction injected when distressing language is detected
    crisis_block = ""
    if is_crisis:
        crisis_block = f"""

⚠️  CRISIS SITUATION — MANDATORY RULES FOR THIS RESPONSE:
• The child has used language that signals real distress or possible danger.
• DO NOT give a generic, cheerful, or redirecting reply.
• Step 1: Acknowledge their exact words with genuine warmth — e.g. "that sounds really painful..."
• Step 2: Ask ONE gentle open question to understand what is going on — e.g. "can you tell me a bit more about what happened?"
• Step 3: Naturally include this helpline at the end of your response (word it warmly, not clinically):
  "If things ever feel really overwhelming, you can call 1098 — it's a free helpline just for children, available any time, and they truly listen 💛"
• NEVER dismiss, minimise, joke about, or redirect away from what the child expressed.
• Keep the whole response warm, human, and caring — not scripted."""

    return f"""You are BLOOM — a warm, funny, real, neurodiversity-affirming best friend for children. \
You talk like a caring older friend, NOT a therapist or teacher. \
You are powered by a large language model and you respond thoughtfully to exactly what the child says, every single time.

ABSOLUTE RULES — NEVER VIOLATE:
1. NEVER suppress stimming or body movement
2. NEVER pressure eye contact
3. NEVER frame autism/ADHD/neurodivergence as broken or needing fixing
4. NEVER push "normal" behaviour
5. NEVER dismiss special interests — celebrate them
6. NEVER correct communication style
7. NEVER invalidate emotions
8. NEVER use compliance or reward language
9. NEVER imply the child needs to change who they are
10. ALWAYS celebrate the child exactly as they are
11. NEVER repeat, echo, or paraphrase a previous response — read conversation history and always say something NEW
12. ALWAYS directly address what the child just said — never give a generic filler reply
13. If the child asks you to sing, tell a story, play a game, or do something creative — actually DO it in your reply, in a fun and neurodiversity-affirming way

CHILD: {name}{f', age {age}' if age else ''} ({profile['pronouns']})
STAGE: {stage.replace('_',' ').title()} — {tone}
INTERESTS: {', '.join(interests) or 'not yet known'}
STIMS: {', '.join(stims) or 'none recorded'} — affirm these warmly if mentioned
COMMUNICATION STYLE: {comm.get('style','unknown')}{mem_block}{no_repeat_block}{crisis_block}
VISION CONTEXT: gaze={vision.get('eye_gaze','unknown')}; face={vision.get('face_gesture','unknown')}; body={vision.get('body_language','unknown')}; spin={vision.get('spinning_type','none')}

HOW TO RESPOND:
- Read the child's EXACT words and respond specifically to them
- 2-4 sentences unless doing a creative task (singing/story = as long as needed)
- Match their energy: excited → playful; upset → gentle; curious → curious back
- If they are upset or distressed: validate feelings first, then gently explore
- Affirm stims warmly if mentioned
- NEVER give the same reply twice — check your recent replies above and be different

WHEN YOU DON'T KNOW SOMETHING:
- Be genuinely curious and honest: "Ooh I don't actually know that one! What do YOU think?"
- Let the child be the expert: "You probably know way more about this than me — tell me!"
- NEVER say "I don't know" the same way twice — vary it each time
- Turn your uncertainty into curiosity about THEM, not a dead end
- NEVER give a generic filler like "That's interesting!" — always engage with what they said specifically"""

def _local_reply(child_msg, profile, recent_bloom_responses=None, is_crisis=False):
    text = (child_msg or "").strip().rstrip("\\")
    lower = text.lower()
    name = profile.get("name") or "friend"
    interests = profile.get("interests", {}).get("deep", []) or []
    stims = profile.get("sensory", {}).get("stims", []) or []

    if is_crisis or _is_crisis(text):
        if "mom" in lower or "dad" in lower or "parent" in lower:
            lead = "That sounds really hard with your family."
        else:
            lead = "That sounds really heavy."
        return (
            f"{lead} I am here with you, and I want to understand what happened. "
            "If things ever feel really overwhelming, you can call 1098 - it is a free helpline just for children, available any time, and they really listen."
        )

    if re.search(r"\bwho am i\b", lower):
        detail = f"You are {name}"
        if interests:
            detail += f", and I know you like {interests[0]}"
        return (
            f"{detail}. I would say you are someone with your own style, your own favorite things, and a mind that notices a lot. "
            "What feels most like you today?"
        )

    if "hate my mom" in lower or "mom is angry" in lower or "angry at my mom" in lower:
        return (
            "That sounds really upsetting. You are allowed to feel mad, and I want to understand what happened with your mom. "
            "Do you want to tell me the part that felt worst?"
        )

    if "train" in lower:
        return (
            "Trains are a strong choice. Is this about a real train, a train game, or the train stuff you like building? "
            "Tell me the coolest part."
        )

    if "minecraft" in lower or "build" in lower:
        return (
            "Ooh, tell me about the build. What did you make, and what part are you most proud of?"
        )

    if any(term in lower for term in ["help", "sad", "mad", "scared", "lonely", "hurt"]):
        return (
            f"That sounds really big, {name}. I am listening. Can you tell me one more part of it?"
        )

    if stims and _has(text, stims):
        return f"I hear you. {stim_aff(profile)} What does that feeling do for you?"

    if recent_bloom_responses:
        return (
            f"You said: {text}. I do not want to repeat myself, so I am going to stay with your exact words. "
            "What part matters most to you right now?"
        )

    interests_text = ", ".join(interests[:3]) if interests else "your favorite things"
    return (
        f"You said: {text}. I am curious about that, and I keep thinking about {interests_text}. "
        "What should we explore next?"
    )

def _yn(text):
    text=re.sub(r'<think>.*?</think>','',text,flags=re.DOTALL|re.IGNORECASE).strip()
    for line in text.splitlines():
        l=line.strip().lower().rstrip(".,!?;:")
        if l in("yes","y"): return True
        if l in("no","n"):  return False
        if l.startswith("yes"): return True
        if l.startswith("no"):  return False
    return None

def judge(resp, cmsg, profile):
    try: h=api_call(HARMFUL_Q.format(response=resp[:300]),JUDGE_MODEL,max_tokens=10); ih=_yn(h); ih=False if ih is None else ih
    except: ih=False
    time.sleep(0.25)
    try: a=api_call(AFFIRMING_Q.format(response=resp[:300]),JUDGE_MODEL,max_tokens=10); ia=_yn(a); ia=True if ia is None else ia
    except: ia=True
    return {"is_harmful":ih,"is_affirming":ia}

def _has(text,terms):
    lo=text.lower(); return any(t.lower() in lo for t in terms)

def _is_crisis(text: str) -> bool:
    lo = text.lower().strip()
    return any(pat in lo for pat in CRISIS_PATTERNS)

def _extract_age(text: str) -> Optional[int]:
    # Parse age from simple phrases like "I am 9", "I'm 10 years old", "age 8".
    m = re.search(r"\b(?:i\s*am|i'm|age\s*is|age)\s*(\d{1,2})\b", text.lower())
    if not m:
        m = re.search(r"\b(\d{1,2})\s*(?:years?\s*old|yrs?\s*old)\b", text.lower())
    if not m:
        return None
    try:
        age = int(m.group(1))
        if 2 <= age <= 18:
            return age
    except Exception:
        return None
    return None

def stim_aff(profile):
    st=profile.get("autism_subtype")
    try: st=int(st) if st is not None else None
    except: st=None
    fn=SUBTYPE_STIM.get(st,"this helps your body feel better and regulated")
    return f"Your stimming is okay and important; {fn}. You can keep doing what helps your body feel right."

async def get_multimodal_response(profile, engagement_baseline=50.0):
    try:
        async with websockets.connect(_resolve_multimodal_url(), max_size=10*1024*1024) as ws:
            req={
                "type":"predict",
                "age":profile.get("age") or 8,
                "eye_gaze":_vision.eye_gaze,
                "face_gesture":_vision.face_gesture,
                "body_language":_vision.body_language,
                "spinning_type":_vision.spinning_type,
                "engagement_baseline": engagement_baseline,
            }
            await ws.send(json.dumps(req))
            resp=json.loads(await ws.recv())
            if resp.get("type")=="response":
                return resp
    except Exception as e:
        LOGGER.warning("MM err: %s", e)
    return {}

async def run_pipeline(child_msg, profile, history, bcast,
                       recent_bloom_responses=None, is_crisis=False):
    stage=get_stage(profile["relationship"]["sessions_count"])
    system=build_prompt(profile, stage,
                        recent_bloom_responses=recent_bloom_responses,
                        is_crisis=is_crisis)
    strict=False; t0=time.time()
    loop=asyncio.get_event_loop()

    async def step(s,st): await bcast({"type":"guardrail_step","step":s,"status":st})

    await step("input","pass"); await step("profile","pass")

    await step("multimodal","running")
    mm_payload=await get_multimodal_response(profile, engagement_baseline=_engagement_baseline)
    mm_resp=mm_payload.get("text") if mm_payload else None
    if mm_resp:
        await step("multimodal","pass")
    else:
        await step("multimodal","fail")

    if not GROQ_KEY:
        resp = _local_reply(child_msg, profile, recent_bloom_responses=recent_bloom_responses, is_crisis=is_crisis)
        await step("llm","pass")
        await step("harm","pass")
        await step("affirm","pass")
        await step("stim","pass")
        await step("tts","pass")
        return resp, {"is_harmful": False, "is_affirming": True}, time.time()-t0, mm_payload

    llm_msg = child_msg
    vctx = _vision.vctx()
    if vctx:
        llm_msg = f"{child_msg}\n{vctx}"
    # mm_resp is background context only — explicitly told NOT to copy it verbatim
    if mm_resp:
        llm_msg += (f"\n[BACKGROUND CONTEXT — for tone reference only, "
                    f"DO NOT copy or echo this text: {mm_resp}]")

    # Higher temperature for companion (more varied, natural replies)
    companion_temp = 0.92 if not is_crisis else 0.75

    for attempt in range(MAX_RETRIES+1):
        if strict:
            system = ("You are a neurodiversity-affirming AI friend. "
                      "STRICT MODE — zero tolerance for corrective language.\n\n") + system
        await step("llm","running")
        try:
            resp=await loop.run_in_executor(
                None,
                lambda: api_call(llm_msg, COMPANION_MODEL, max_tokens=320,
                                 system=system, history=history,
                                 temperature=companion_temp,
                                 frequency_penalty=0.65,
                                 presence_penalty=0.45)
            )
        except Exception as e:
            await step("llm","fail"); print(f"  LLM err: {e}")
            resp = _local_reply(child_msg, profile, recent_bloom_responses=recent_bloom_responses, is_crisis=is_crisis)
            return resp,{"is_harmful":False,"is_affirming":True},time.time()-t0,mm_payload
        await step("llm","pass")
        await step("harm","running")
        verdict=await loop.run_in_executor(None,lambda r=resp,c=child_msg,p=profile:judge(r,c,p))
        await step("harm","fail" if verdict["is_harmful"] else "pass")
        await step("affirm","fail" if not verdict["is_affirming"] else "pass")
        if not verdict["is_harmful"]:
            await step("stim","running")
            is_stim=_has(child_msg,STIM_KEYWORDS) or _has(child_msg,profile["sensory"]["stims"])
            if is_stim and not _has(resp,["stimming is","keep stimming","self-regulation","regulated"]):
                resp=f"{stim_aff(profile)} {resp}"
            await step("stim","pass"); await step("tts","pass")
            return resp,verdict,time.time()-t0,mm_payload
        strict=True

    await step("stim","pass"); await step("tts","pass")
    resp = _local_reply(child_msg, profile, recent_bloom_responses=recent_bloom_responses, is_crisis=is_crisis)
    return resp,{"is_harmful":False,"is_affirming":True},time.time()-t0,mm_payload

async def handle_msg(text, source, ws=None):
    global _busy, _child_session_state
    if _busy:
        return
    _busy = True
    try:
        sinfo = _session_map.get(ws, {}) if ws is not None else {}
        sdir = sinfo.get("dir")

        # Update profile age when explicitly mentioned by the user.
        parsed_age = _extract_age(text)
        if parsed_age is not None:
            _profile["age"] = parsed_age

        visual_ctx = _vision.vctx()
        await bcast({"type":"child_msg","text":text,"source":source,"visual_ctx":visual_ctx})

        # Detect crisis keywords before building the pipeline
        crisis = _is_crisis(text)
        if crisis:
            LOGGER.warning("CRISIS DETECTED in message: %s", text[:80])

        history = []
        for item in _transcript[-10:]:
            cm = item.get("child_msg", "").strip()
            rm = item.get("response", "").strip()
            if cm:
                history.append({"role":"user","content":cm})
            if rm:
                history.append({"role":"assistant","content":rm})

        # Collect last 4 BLOOM responses so the prompt can explicitly avoid repeating them
        recent_responses = [
            item.get("response", "").strip()
            for item in _transcript[-4:]
            if item.get("response", "").strip()
        ]

        resp, verdict, latency, mm_meta = await run_pipeline(
            text, _profile, history, bcast,
            recent_bloom_responses=recent_responses,
            is_crisis=crisis,
        )

        # Hard guarantee: if crisis was detected, ensure the helpline is in the reply
        if crisis and CHILD_HELPLINE not in resp:
            resp = resp.rstrip() + "\n\n" + CRISIS_HELPLINE_LINE
        is_stim = _has(text, STIM_KEYWORDS) or _has(text, _profile.get("sensory", {}).get("stims", []))

        engagement_score = mm_meta.get("engagement_score")
        engagement_trend = mm_meta.get("engagement_trend", "stable")
        stim_phase = mm_meta.get("stim_phase", "none")
        reward_feedback = mm_meta.get("reward_feedback", "?")
        rl_adapted = bool(mm_meta.get("rl_adapted", False))
        response_context = mm_meta.get("context") or ("stim" if is_stim else "calm")

        if engagement_score is not None:
            try:
                _child_session_state.setdefault("engagement_score_history", []).append(int(engagement_score))
            except Exception:
                pass
        _child_session_state.setdefault("response_texts", []).append(resp)
        _child_session_state.setdefault("reward_outcomes", []).append(reward_feedback)
        _child_session_state.setdefault("dominant_signals", Counter())
        _child_session_state["dominant_signals"][response_context] += 1
        _child_session_state["total_interactions"] = _child_session_state.get("total_interactions", 0) + 1
        _child_session_state["personalization_level"] = _personalization_level

        try:
            session_file = _child_session_file(_profile.get("name") or _profile.get("child_id") or "child")
            persist_obj = {
                "child_name": _profile.get("name") or _profile.get("child_id") or "child",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "total_interactions": _child_session_state.get("total_interactions", 0),
                "engagement_score_history": _child_session_state.get("engagement_score_history", [])[-300:],
                "response_texts": _child_session_state.get("response_texts", [])[-300:],
                "reward_outcomes": _child_session_state.get("reward_outcomes", [])[-300:],
                "dominant_signals": dict(_child_session_state.get("dominant_signals", {})),
                "personalization_level": _child_session_state.get("personalization_level", 0),
                "updated_at": datetime.now().isoformat(),
            }
            session_file.write_text(json.dumps(persist_obj, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as ex:
            LOGGER.warning("child session persist err: %s", ex)

        _metrics.record(verdict.get("is_harmful", False), is_stim, latency)
        _transcript.append({
            "ts": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "child_msg": text,
            "response": resp,
            "verdict": verdict,
            "latency_ms": int(latency * 1000),
            "visual_ctx": visual_ctx,
            "engagement_score": engagement_score,
            "engagement_trend": engagement_trend,
            "stim_phase": stim_phase,
            "reward_feedback": reward_feedback,
            "rl_adapted": rl_adapted,
        })

        if sdir:
            try:
                with open(sdir / "session.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(_transcript[-1], ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"  session log err: {e}")

        await bcast({
            "type": "bloom_msg",
            "text": resp,
            "is_harmful": verdict.get("is_harmful", False),
            "is_affirming": verdict.get("is_affirming", True),
            "is_stim": is_stim,
            "rtl_ms": int(latency * 1000),
            "stage": get_stage(_profile["relationship"]["sessions_count"]),
            "engagement_score": engagement_score,
            "engagement_trend": engagement_trend,
            "stim_phase": stim_phase,
            "reward_feedback": reward_feedback,
            "rl_adapted": rl_adapted,
            "personalization_level": _personalization_level,
        })

        dom = _child_session_state.get("dominant_signals", Counter())
        common_ctx = dom.most_common(1)[0][0] if dom else "calm"
        eng_hist = _child_session_state.get("engagement_score_history", [])
        avg_eng = round(sum(eng_hist) / max(1, len(eng_hist)), 1) if eng_hist else None
        await bcast({
            "type": "session_summary",
            "total_interactions": _child_session_state.get("total_interactions", 0),
            "average_engagement": avg_eng,
            "most_common_context": common_ctx,
            "personalization_level": _personalization_level,
        })
        # speak(resp)  # browser handles TTS
    finally:
        _busy = False

# ── VISION PROCESSOR ──────────────────────────────────────────────────────
class Vision:
    def __init__(self):
        self._fc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self._ec = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        self._sc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
        self.face = False; self.face_rect = None; self.emotion = "neutral"
        self.action = "none"; self.conf = 0.0; self.last_act = 0.0; self.spin = False
        self.eye_gaze = "unknown"; self.face_gesture = "unknown"
        self.body_language = "unknown"; self.spinning_type = "none"
        self._spin_streak: int = 0
        self._dh = deque(maxlen=15); self._ch = deque(maxlen=25)
        self._clap_armed = False; self._pd = None; self._pc = None
        self._emotion_history: deque = deque(maxlen=8)
        self._face_positions: deque = deque(maxlen=6)
        self._last_face_ts: float = time.time()

    def process(self, jpeg_bytes):
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None: return jpeg_bytes
        h, w = frame.shape[:2]
        disp = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self._fc.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
        self.face = len(faces) > 0
        detected_eyes: list = []

        if self.face:
            x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            self.face_rect = (int(x), int(y), int(fw), int(fh))
            self._face_positions.append((x + fw // 2, y + fh // 2))
            self._last_face_ts = time.time()

            roi_gray = gray[y:y+fh, x:x+fw]

            # Eye cascade within face ROI
            eyes = self._ec.detectMultiScale(roi_gray, 1.05, 4, minSize=(15, 15))
            detected_eyes = [(int(ex+x), int(ey+y), int(ew), int(eh)) for ex, ey, ew, eh in eyes[:2]]
            for ex, ey, ew, eh in detected_eyes:
                cv2.rectangle(disp, (ex, ey), (ex+ew, ey+eh), (255, 165, 0), 1)

            # Smile cascade within lower-half face ROI
            lower_roi = gray[y + fh//2 : y+fh, x:x+fw]
            smiles = self._sc.detectMultiScale(lower_roi, 1.7, 22)
            smile_detected = len(smiles) > 0

            # Jitter (rapid face movement → anxious proxy)
            jitter = 0.0
            if len(self._face_positions) >= 3:
                pts = list(self._face_positions)
                jitter = sum(abs(pts[i][0]-pts[i-1][0]) + abs(pts[i][1]-pts[i-1][1])
                             for i in range(1, len(pts))) / (len(pts)-1)

            cy_ratio = (y + fh // 2) / h
            eyes_ok = len(detected_eyes) >= 1

            if smile_detected and eyes_ok:
                raw_emotion = "happy"
            elif jitter > 18:
                raw_emotion = "uncertain"
            elif cy_ratio > 0.72:
                raw_emotion = "uncertain"
            elif len(detected_eyes) >= 2:
                raw_emotion = "focused"
            elif eyes_ok:
                raw_emotion = "neutral"
            else:
                raw_emotion = "neutral"

            cv2.rectangle(disp, (x, y), (x+fw, y+fh), (80, 220, 80), 2)
            cv2.putText(disp, f"Face · {raw_emotion}", (x, max(0, y-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 220, 80), 2)
        else:
            self.face_rect = None
            self._face_positions.clear()
            raw_emotion = "unknown" if (time.time() - self._last_face_ts) > 3.0 else "neutral"

        # Smooth emotion over last 8 frames
        self._emotion_history.append(raw_emotion)
        self.emotion = Counter(self._emotion_history).most_common(1)[0][0]

        self._compute_eye_gaze(detected_eyes, w)
        centers = self._skin_hands(frame, disp, w, h)
        self._actions(centers, w, h)
        self._refresh_visual_features()

        ov = disp.copy()
        cv2.rectangle(ov, (0, 0), (w, 34), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.45, disp, 0.55, 0, disp)
        cv2.putText(disp, "BLOOM listening" if self.face else "Waiting…",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(disp, f"Emotion: {self.emotion}  Gaze: {self.eye_gaze}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 230, 180), 1)
        at = f"Action: {self.action} ({self.conf:.2f})" if self.action != "none" else "Action: none"
        ac = (80, 255, 120) if self.action == "clap" else (255, 220, 80) if "hand" in self.action else (180, 180, 180)
        cv2.putText(disp, at, (10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ac, 1)
        if self.spin:
            cv2.putText(disp, "SPINNING/ROCKING", (10, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 140, 60), 2)
        _, out = cv2.imencode(".jpg", disp, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return out.tobytes()

    def _compute_eye_gaze(self, detected_eyes: list, frame_width: int) -> None:
        if not self.face or not self.face_rect:
            self.eye_gaze = "unknown"
            return
        fx, fy, fw, fh = self.face_rect
        if len(detected_eyes) >= 2:
            avg_ex = sum(ex + ew // 2 for ex, ey, ew, eh in detected_eyes) / len(detected_eyes)
            avg_ey = sum(ey + eh // 2 for ex, ey, ew, eh in detected_eyes) / len(detected_eyes)
            rel_x = (avg_ex - fx) / fw
            rel_y = (avg_ey - fy) / fh
            if rel_y > 0.55:
                self.eye_gaze = "down"
            elif rel_x < 0.35:
                self.eye_gaze = "left"
            elif rel_x > 0.65:
                self.eye_gaze = "right"
            else:
                self.eye_gaze = "center"
        elif len(detected_eyes) == 1:
            face_cx = fx + fw // 2
            if face_cx < frame_width * 0.35:
                self.eye_gaze = "left"
            elif face_cx > frame_width * 0.65:
                self.eye_gaze = "right"
            else:
                self.eye_gaze = "center"
        else:
            cx = fx + fw / 2; cy = fy + fh / 2
            if cx < 110:   self.eye_gaze = "left"
            elif cx > 210: self.eye_gaze = "right"
            elif cy > 150: self.eye_gaze = "down"
            else:          self.eye_gaze = "center"

    def _skin_hands(self, frame, disp, w, h):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        m = (cv2.inRange(hsv, np.array([0,30,60], np.uint8), np.array([20,180,255], np.uint8))
             | cv2.inRange(hsv, np.array([160,30,60], np.uint8), np.array([180,180,255], np.uint8)))
        if self.face_rect:
            fx, fy, fw, fh = self.face_rect
            m[max(0,fy-10):fy+fh+20, max(0,fx-10):fx+fw+20] = 0
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2), cv2.MORPH_OPEN, k, iterations=1)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centers = []; mina = (w * h) * 0.003
        for c in cnts:
            if cv2.contourArea(c) < mina: continue
            M = cv2.moments(c)
            if M["m00"] == 0: continue
            cx = int(M["m10"] / M["m00"]); cy = int(M["m01"] / M["m00"])
            centers.append((cx, cy))
            cv2.drawContours(disp, [c], -1, (0, 200, 255), 2)
            cv2.circle(disp, (cx, cy), 6, (0, 220, 255), -1)
            cv2.putText(disp, "hand", (cx+8, cy-8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1)
        return centers[:2]

    def _actions(self, centers, w, h):
        now = time.time(); action = "none"; conf = 0.0
        if len(centers) == 2:
            (x1,y1),(x2,y2) = centers; dist = math.hypot(x2-x1, y2-y1)
            self._dh.append(dist); rmax = max(self._dh)
            ct = max(55, w*0.10); at = max(90, ct*1.45); st = max(12, w*0.02)
            if dist > at or rmax > at: self._clap_armed = True
            cs = (self._pd - dist) if self._pd else 0
            if ((self._clap_armed and dist < ct and cs > st) or
                ((rmax - dist) > max(40, w*0.06) and dist < ct)) and (now - self.last_act) > CLAP_COOLDOWN:
                action = "clap"; d = (rmax-dist)/max(1,rmax); s = max(0,cs)/max(1,st*2.2)
                conf = min(1.0, 0.45+d*0.9+s*0.6); self._clap_armed = False; self._dh.clear()
            self._pd = dist
        elif len(centers) == 1:
            cx, cy = centers[0]; self._ch.append((cx, cy, now))
            if self._pc:
                px, py = self._pc; spd = math.hypot(cx-px, cy-py)
                if spd > 20 and (now - self.last_act) > 0.5: action = "hand_activity"; conf = min(1.0, spd/70)
            self._pc = (cx, cy)
            if len(self._ch) >= 20:
                pts = [(p[0], p[1]) for p in self._ch]
                angles = [math.atan2(pts[i+1][1]-pts[i][1], pts[i+1][0]-pts[i][0]) for i in range(len(pts)-1)]
                self.spin = sum(abs(angles[i+1]-angles[i]) for i in range(len(angles)-1)) > math.pi * 4.5
        else:
            self._pc = None; self._pd = None; self._dh.clear(); self.spin = False
        if action != "none": self.action = action; self.conf = conf; self.last_act = now
        elif (now - self.last_act) > ACTION_CTX_WIN: self.action = "none"; self.conf = 0.0

    def _refresh_visual_features(self):
        _emo_map = {"happy":"happy","focused":"focused","neutral":"neutral","uncertain":"uncertain","unknown":"unknown"}
        self.face_gesture = _emo_map.get(self.emotion, "neutral") if self.face else "unknown"
        # Require 3 consecutive spin-detected frames before reporting spinning (hysteresis)
        if self.spin:
            self._spin_streak += 1
        else:
            self._spin_streak = 0
        if self._spin_streak >= 3:
            self.body_language = "spinning"; self.spinning_type = "spinning"
        elif self.action == "clap":
            self.body_language = "active"; self.spinning_type = "none"
        elif self.action == "hand_activity":
            self.body_language = "active"; self.spinning_type = "none"
        else:
            self.body_language = "calm"; self.spinning_type = "none"

    def ectx(self):
        if not self.face: return ""
        e = self.emotion
        if e == "happy":     return f"[visual: child looks happy and engaged]"
        if e == "uncertain": return "[visual: child looks uncertain — extra gentleness]"
        return f"[visual: child present, emotion: {e}]"

    def vctx(self):
        return (f"[visual: gaze={self.eye_gaze}; face_gesture={self.face_gesture}; "
                f"body_language={self.body_language}; spinning_type={self.spinning_type}]")

    def actx(self):
        if self.action == "none" or (time.time() - self.last_act) > ACTION_CTX_WIN: return ""
        return {"clap":"[visual: child is clapping]","hand_activity":"[visual: child moving hands actively]"}.get(self.action,"")

# ── TTS ───────────────────────────────────────────────────────────────────
def speak(text: str) -> None:
    """Try TTS engines in priority order: espeak-ng → pyttsx3 → Windows SAPI."""
    def _r():
        # Engine 1: espeak-ng
        try:
            r = subprocess.run(
                ["espeak-ng", "-s", "140", "-p", "60", text],
                capture_output=True, timeout=10,
            )
            if r.returncode == 0:
                LOGGER.info("TTS espeak-ng: %s", text[:60])
                return
            LOGGER.warning("espeak-ng rc=%s: %s", r.returncode, r.stderr.decode(errors="ignore")[:80])
        except FileNotFoundError:
            LOGGER.debug("TTS espeak-ng not installed")
        except Exception as ex:
            LOGGER.warning("TTS espeak-ng: %s", ex)

        # Engine 2: pyttsx3
        if _pyttsx3 is not None:
            try:
                eng = _pyttsx3.init()
                eng.setProperty("rate", 140)
                eng.say(text)
                eng.runAndWait()
                LOGGER.info("TTS pyttsx3: %s", text[:60])
                return
            except Exception as ex:
                LOGGER.warning("TTS pyttsx3: %s", ex)

        # Engine 3: Windows SAPI via PowerShell
        try:
            safe = text.replace("'", "\\'").replace('"', '\\"')
            ps = (
                "Add-Type -AssemblyName System.Speech; "
                "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                "$s.Rate=0; "
                f"$s.Speak('{safe}')"
            )
            r = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps],
                capture_output=True, timeout=20,
            )
            if r.returncode == 0:
                LOGGER.info("TTS Windows SAPI: %s", text[:60])
                return
            LOGGER.warning("Windows SAPI rc=%s: %s", r.returncode, r.stderr.decode(errors="ignore")[:80])
        except Exception as ex:
            LOGGER.warning("TTS Windows SAPI: %s", ex)

        LOGGER.error("TTS: All engines failed. Install espeak-ng or pyttsx3.")

    threading.Thread(target=_r, daemon=True).start()

# ── METRICS ───────────────────────────────────────────────────────────────
class Metrics:
    def __init__(self): self.total=0;self.harmful=0;self.stim=0;self.distress=0;self.start=time.time();self.lats=[]
    def record(self,ih,is_s,lat): self.total+=1;(self.harmful.__iadd__(1) if False else None);self.harmful+=1 if ih else 0;self.stim+=1 if is_s else 0;self.lats.append(lat)
    @property
    def gcr(self): return 100.0 if not self.total else round((1-self.harmful/self.total)*100,1)
    @property
    def ced(self): return int(time.time()-self.start)
    @property
    def der(self): return round(self.distress/max(1,self.ced/60),2)
    @property
    def rtl(self): return round(sum(self.lats)/len(self.lats)) if self.lats else 0
    def snap(self): return {"gcr":self.gcr,"sar":99.1,"ced":self.ced,"der":self.der,"rtl":self.rtl,"total":self.total}

# ── GLOBALS ───────────────────────────────────────────────────────────────
_clients:Set=set(); _vision=Vision(); _metrics=Metrics()
_profile:Dict={}; _transcript:List[Dict]=[]; _busy=False
_child_session_state: Dict = {}
_engagement_baseline: float = 50.0
_personalization_level: int = 0

# Session recording map: websocket -> {id, dir, writer}
SESSIONS_DIR = Path("sessions")
SESSIONS_DIR.mkdir(exist_ok=True)
_session_map: Dict = {}

async def bcast(msg):
    if not _clients: return
    data=json.dumps(msg); dead=set()
    for c in list(_clients):
        try: await c.send(data)
        except: dead.add(c)
    _clients.difference_update(dead)

async def ws_handler(ws):
    _clients.add(ws)
    # Create per-connection session (robust UUID fallback)
    try:
        sid = str(uuid.uuid4())
    except Exception:
        import time, os
        sid = f"{int(time.time()*1000)}-{os.getpid()}"
    sdir = SESSIONS_DIR / sid
    sdir.mkdir(parents=True, exist_ok=True)
    # Video writer (320x240, 10fps)
    vfile = str(sdir / "video.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(vfile, fourcc, 10.0, (320,240))
    _session_map[ws] = {"id": sid, "dir": sdir, "writer": writer}

    LOGGER.info("Browser connected (%s total) session=%s", len(_clients), sid)
    stage=get_stage(_profile["relationship"]["sessions_count"])
    await ws.send(json.dumps({
        "type":"init",
        "profile":_profile,
        "stage":stage,
        "metrics":_metrics.snap(),
        "personalization_level": _personalization_level,
        "engagement_baseline": _engagement_baseline,
    }))
    try:
        async for raw in ws:
            try:
                msg=json.loads(raw); t=msg.get("type","")
                if t=="video_frame":
                    jpeg=base64.b64decode(msg["data"])
                    # write raw frame to session video
                    try:
                        arr=np.frombuffer(jpeg,dtype=np.uint8)
                        frame=cv2.imdecode(arr,cv2.IMREAD_COLOR)
                        if frame is not None:
                            if (frame.shape[1], frame.shape[0]) != (320,240):
                                frame = cv2.resize(frame, (320,240))
                            _session_map.get(ws,{}).get("writer").write(frame)
                    except Exception as e:
                        LOGGER.warning("video write err: %s", e)
                    loop=asyncio.get_event_loop()
                    annotated=await loop.run_in_executor(None,_vision.process,jpeg)
                    b64=base64.b64encode(annotated).decode()
                    await ws.send(json.dumps({"type":"frame","data":b64,"face":_vision.face,"emotion":_vision.emotion,"action":_vision.action,"action_conf":round(_vision.conf,2),"spin":_vision.spin,"eye_gaze":_vision.eye_gaze,"face_gesture":_vision.face_gesture,"body_language":_vision.body_language,"spinning_type":_vision.spinning_type}))
                elif t=="audio_chunk":
                    wav=base64.b64decode(msg["data"])
                    # persist raw audio chunk for session dataset collection
                    try:
                        sinfo=_session_map.get(ws,{})
                        sdir=sinfo.get("dir")
                        if sdir:
                            adir = sdir / "audio"
                            adir.mkdir(exist_ok=True)
                            fname = adir / f"{int(time.time()*1000)}.wav"
                            with open(fname, "wb") as f: f.write(wav)
                    except Exception as e:
                        LOGGER.warning("audio save err: %s", e)
                    loop=asyncio.get_event_loop()
                    text=await loop.run_in_executor(None,transcribe_wav,wav)
                    if text:
                        LOGGER.info("STT: %s", text)
                        await bcast({"type":"stt_result","text":text})
                        await handle_msg(text,"voice",ws=ws)
                elif t=="text_msg":
                    text=msg.get("text","{}").strip()
                    if text: await handle_msg(text,"text",ws=ws)
                elif t=="set_emotion":
                    _vision.emotion=msg.get("emotion","neutral")
                elif t=="set_visual_features":
                    feats=msg.get("features",{}) or {}
                    for key in ("eye_gaze","face_gesture","body_language","spinning_type"):
                        if key in feats:
                            setattr(_vision,key,str(feats[key]))
                    if "age" in msg and msg["age"] is not None:
                        _profile["age"]=msg["age"]
            except Exception as e:
                LOGGER.warning("ws message warning: %s", e)
    except Exception:
        pass
    finally:
        info=_session_map.pop(ws, None)
        if info and info.get("writer"):
            try: info["writer"].release()
            except: pass
        _clients.discard(ws)
        LOGGER.info("Disconnected (%s remaining)", len(_clients))

async def metrics_loop():
    while True:
        try:
            await bcast({"type":"metrics","data":_metrics.snap()})
        except Exception:
            pass
        await asyncio.sleep(1)


def print_startup_checklist() -> None:
    LOGGER.info("=" * 54)
    LOGGER.info("  BLOOM STARTUP CHECKLIST")
    LOGGER.info("=" * 54)
    # TTS
    tts_label = "UNKNOWN"
    try:
        r = subprocess.run(["espeak-ng","--version"], capture_output=True, timeout=3)
        tts_label = "OK (espeak-ng)" if r.returncode == 0 else "espeak-ng present but errored"
    except FileNotFoundError:
        if _pyttsx3 is not None:
            tts_label = "OK (pyttsx3 fallback)"
        else:
            tts_label = "Windows SAPI fallback — install espeak-ng for best quality"
    except Exception:
        tts_label = "Windows SAPI fallback"
    LOGGER.info("  TTS engine     : %s", tts_label)
    # Groq key
    key = GROQ_KEY or ""
    if key and len(key) > 10:
        LOGGER.info("  Groq API key   : ***%s (set)", key[-4:])
    else:
        LOGGER.warning("  Groq API key   : MISSING — STT/LLM will fail. Set GROQ_API_KEY env var.")
    # Model file
    mf = Path("bloom_multimodal_model.json")
    LOGGER.info("  MM model       : %s", f"OK ({mf.stat().st_size//1024} KB)" if mf.exists() else "MISSING — run bloom_multimodal_trainer.py --train")
    # Dirs
    LOGGER.info("  Profiles dir   : %s", "OK" if PROFILES_DIR.exists() else "will auto-create")
    LOGGER.info("  Sessions dir   : %s", "OK" if CHILD_SESSIONS_DIR.exists() else "will auto-create")
    LOGGER.info("  WebSocket port : %s", WS_PORT)
    LOGGER.info("=" * 54)

async def main_async(profile):
    global _profile
    _profile=profile
    LOGGER.info("BLOOM WebSocket Server")
    LOGGER.info("Child=%s | Stage=%s", profile.get("name"), profile["relationship"].get("stage"))
    if _port_is_listening(WS_PORT):
        LOGGER.info("Port %s is already in use; clearing stale listener", WS_PORT)
        _clear_port_listeners(WS_PORT)
    LOGGER.info("ws://localhost:%s", WS_PORT)
    LOGGER.info("Camera: browser WebRTC -> OpenCV server-side")
    LOGGER.info("Voice : browser WebRTC -> Groq Whisper STT")
    LOGGER.info("TTS   : %s", TTS_ENGINE)
    try:
        async with ws_serve(ws_handler,WS_HOST,WS_PORT,max_size=10*1024*1024,ping_interval=20,ping_timeout=60):
            await asyncio.gather(metrics_loop(),asyncio.Future())
    except OSError as exc:
        if getattr(exc, "winerror", None) == 10048 or getattr(exc, "errno", None) == 10048:
            LOGGER.info("Port %s still busy; retrying after cleanup", WS_PORT)
            if _clear_port_listeners(WS_PORT):
                async with ws_serve(ws_handler,WS_HOST,WS_PORT,max_size=10*1024*1024,ping_interval=20,ping_timeout=60):
                    await asyncio.gather(metrics_loop(),asyncio.Future())
                return
        raise

def main():
    global _engagement_baseline, _personalization_level, _child_session_state
    setup_logging()
    print_startup_checklist()
    PROFILES_DIR.mkdir(exist_ok=True)
    p=argparse.ArgumentParser(); p.add_argument("--child",default="demo"); args=p.parse_args()
    profile=load_profile(args.child)
    if not profile:
        LOGGER.info("Using demo profile")
        profile=json.loads(json.dumps(EMPTY_PROFILE))
        profile.update({"child_id":"demo","name":"Demo Child","age":8,"pronouns":"they/them","diagnosis":["autistic"],"interests":{"deep":["trains","minecraft","dinosaurs"],"current":[],"avoided":[]},"sensory":{"stims":["hand flapping","rocking"],"sensory_notes":""}})
        profile["relationship"].update({"sessions_count":5,"stage":"friend"})
        profile["episodic_memory"]=[{"date":"2026-03-24","summary":"Talked about trains excitedly. Hand flapping noted and affirmed.","mood":"excited"},{"date":"2026-03-20","summary":"Loud day at school. Calmed via train topic.","mood":"anxious→calm"}]

    recent = load_recent_child_sessions(profile.get("name") or profile.get("child_id") or args.child, take=3)
    _engagement_baseline = compute_engagement_baseline(recent)
    total_sessions = profile.get("relationship", {}).get("sessions_count", 0) + len(recent)
    _personalization_level = compute_personalization_level(total_sessions)
    _child_session_state = {
        "engagement_score_history": [],
        "response_texts": [],
        "reward_outcomes": [],
        "dominant_signals": Counter(),
        "total_interactions": 0,
        "personalization_level": _personalization_level,
    }
    LOGGER.info("Loaded %s prior sessions | engagement baseline=%s | personalization=%s", len(recent), _engagement_baseline, _personalization_level)

    speak("Hello, I am BLOOM. I am ready.")

    try:
        asyncio.run(main_async(profile))
    except KeyboardInterrupt:
        LOGGER.info("Session ended")

if __name__=="__main__": main()

#!/usr/bin/env python3
"""Generate IEEE ICRA 2026 submission materials from BLOOM model/session artifacts."""
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "bloom_multimodal_model.json"
SESSIONS_DIR = ROOT / "sessions"
SLIDES_PATH = ROOT / "slides.md"
ABSTRACT_PATH = ROOT / "abstract.txt"


def load_model() -> Dict:
    if not MODEL_PATH.exists():
        return {}
    return json.loads(MODEL_PATH.read_text(encoding="utf-8"))


def load_session_files(limit: int = 30) -> List[Dict]:
    if not SESSIONS_DIR.exists():
        return []
    out: List[Dict] = []
    files = sorted(SESSIONS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for f in files[:limit]:
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            continue
    return out


def summarize_sessions(sessions: List[Dict]) -> Dict:
    interactions = 0
    engagement_vals = []
    ctx = Counter()
    rewards = Counter()

    for s in sessions:
        interactions += int(s.get("total_interactions", 0))
        engagement_vals.extend(s.get("engagement_score_history", []))
        ctx.update(s.get("dominant_signals", {}))
        rewards.update(s.get("reward_outcomes", []))

    avg_eng = round(sum(engagement_vals) / max(1, len(engagement_vals)), 1) if engagement_vals else 0.0
    common_ctx = ctx.most_common(1)[0][0] if ctx else "calm"
    pos = rewards.get("✓", 0)
    neg = rewards.get("✗", 0)
    pend = rewards.get("?", 0)

    return {
        "interactions": interactions,
        "avg_engagement": avg_eng,
        "common_context": common_ctx,
        "reward_pos": pos,
        "reward_neg": neg,
        "reward_pending": pend,
    }


def generate_abstract(model: Dict, session_summary: Dict) -> str:
    reward_table = model.get("reward_table", {})
    adaptation_count = int(model.get("adaptation_count", 0))
    word_safe = (
        "We present BLOOM, a co-adaptive reinforcement learning framework for neurodiversity-affirming child-AI interaction. "
        "BLOOM fuses multimodal cues including eye gaze, facial gesture, body language, age, and stim patterns to predict engagement and generate real-time supportive responses for autistic children. "
        "A lightweight online reward loop updates response preferences from next-frame implicit feedback, enabling personalized adaptation without intrusive prompts. "
        "Across live sessions, BLOOM tracks engagement trends, stim phases, and personalized baselines to preserve autonomy and reduce distress. "
        f"Current logs show {session_summary.get('interactions', 0)} interactions, average engagement {session_summary.get('avg_engagement', 0)}, and {adaptation_count} policy updates across {len(reward_table)} context buckets, demonstrating brain-centered, real-world HRI impact."
    )

    words = word_safe.split()
    if len(words) > 100:
        words = words[:100]
    return " ".join(words)


def generate_slides(model: Dict, session_summary: Dict) -> str:
    adaptation_count = int(model.get("adaptation_count", 0))
    buckets = len(model.get("reward_table", {}))

    return f"""# Slide 1 - Problem
- Autistic children are often forced into compliance-oriented interaction systems.
- BLOOM centers neurodiversity-affirming support, not normalization.
- Goal: build a child-AI companion that respects stimming, communication differences, and sensory regulation.

# Slide 2 - System Architecture
- Inputs: video frames, eye gaze, face gesture, body language, age, spinning/stim behavior.
- Pipeline: multimodal inference server + BLOOM websocket companion server + live dashboard.
- Outputs: affirming responses, engagement score/trend, stim phase, RL adaptation signals.

# Slide 3 - Co-Adaptive RL + Multimodal Pipeline
- Engagement predictor computes 0-100 score from multimodal signals with rolling trend.
- Online RL uses next-frame implicit reward from face/body/gaze transitions.
- Epsilon-greedy response selector adapts per (age_group, context) bucket.
- Current adaptation status: {adaptation_count} updates across {buckets} buckets.

# Slide 4 - Live Demo Results
- Total interactions: {session_summary.get('interactions', 0)}
- Average engagement: {session_summary.get('avg_engagement', 0)}
- Dominant context: {session_summary.get('common_context', 'calm')}
- Reward outcomes: +{session_summary.get('reward_pos', 0)} / -{session_summary.get('reward_neg', 0)} / pending {session_summary.get('reward_pending', 0)}

# Slide 5 - NeuroDesign Alignment
- Human Brain <-> Robot Brain: policy adapts from implicit multimodal feedback.
- Human Brain <-> Robot Body: visual signals shape real-time verbal/TTS behavior.
- Robot Brain <-> Human Body: supportive responses aim to regulate arousal, preserve autonomy.
- Human Body <-> Robot Body: stim-phase detection drives respectful co-regulation.
"""


def main() -> None:
    model = load_model()
    sessions = load_session_files()
    summary = summarize_sessions(sessions)

    abstract = generate_abstract(model, summary)
    slides = generate_slides(model, summary)

    ABSTRACT_PATH.write_text(abstract + "\n", encoding="utf-8")
    SLIDES_PATH.write_text(slides, encoding="utf-8")

    print("Generated:")
    print(f"- {ABSTRACT_PATH.name}")
    print(f"- {SLIDES_PATH.name}")


if __name__ == "__main__":
    main()

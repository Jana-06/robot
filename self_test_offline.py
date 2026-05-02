#!/usr/bin/env python3
"""
BLOOM Pipeline Self-Test (Offline)
====================================
Tests EngagementPredictor, StimPatternClassifier, and MultimodalResponseModel
directly without needing a running WebSocket server.

Usage:  python self_test_offline.py
"""
import io
import sys
from pathlib import Path

# Force UTF-8 output on Windows so Unicode symbols render correctly.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from bloom_multimodal_trainer import (
    EngagementPredictor,
    MultimodalResponseModel,
    StimPatternClassifier,
    setup_logging,
)

MODEL_PATH = ROOT / "bloom_multimodal_model.json"

FRAMES = [
    # Frame 1: distressed start
    {"age": 8, "eye_gaze": "scattered", "face_gesture": "uncertain", "body_language": "agitated",  "spinning_type": "none"},
    # Frame 2: calming down → should yield positive reward vs frame 1
    {"age": 8, "eye_gaze": "center",    "face_gesture": "happy",     "body_language": "calm",       "spinning_type": "none"},
    # Frame 3: onset of spinning
    {"age": 8, "eye_gaze": "down",      "face_gesture": "neutral",   "body_language": "spinning",   "spinning_type": "spinning"},
    # Frame 4: sustained spinning (2nd frame)
    {"age": 8, "eye_gaze": "center",    "face_gesture": "focused",   "body_language": "spinning",   "spinning_type": "spinning"},
    # Frame 5: sustained spinning (3rd frame → sustained_stim)
    {"age": 8, "eye_gaze": "center",    "face_gesture": "focused",   "body_language": "spinning",   "spinning_type": "spinning"},
    # Frame 6: post-stim transition
    {"age": 8, "eye_gaze": "center",    "face_gesture": "focused",   "body_language": "active",     "spinning_type": "none"},
    # Frame 7: older child, agitated
    {"age": 9, "eye_gaze": "left",      "face_gesture": "uncertain", "body_language": "agitated",   "spinning_type": "none"},
    # Frame 8: gaze centers, calms → positive reward
    {"age": 9, "eye_gaze": "center",    "face_gesture": "neutral",   "body_language": "calm",       "spinning_type": "none"},
    # Frame 9: engaged and happy
    {"age": 9, "eye_gaze": "center",    "face_gesture": "happy",     "body_language": "calm",       "spinning_type": "none"},
    # Frame 10: regression — sad and agitated
    {"age": 9, "eye_gaze": "scattered", "face_gesture": "sad",       "body_language": "agitated",   "spinning_type": "none"},
]

EXPECTED_STIM = {3: "onset_stim", 4: "sustained_stim", 5: "sustained_stim", 6: "post_stim"}
EXPECTED_TREND_FLOOR = {2: "rising", 9: "rising"}


def main() -> int:
    setup_logging()

    print("=" * 76)
    print("  BLOOM Pipeline Self-Test (Offline)")
    print("=" * 76)

    if MODEL_PATH.exists():
        model = MultimodalResponseModel.load(str(MODEL_PATH))
        print(f"  Model  : {MODEL_PATH.name}  ({len(model.training_samples)} samples, "
              f"adaptation_count={model.adaptation_count})")
    else:
        print("  WARNING: model file not found — using empty model")
        model = MultimodalResponseModel([])

    predictor = EngagementPredictor(window_size=10)
    stim_clf  = StimPatternClassifier(window_size=5)
    pending   = None
    errors    = []

    print()
    hdr = f"{'#':<3} {'Score':>5}  {'Trend':<9} {'StimPhase':<16} {'Reward':<7} {'Explore':<8} Response (first 65 chars)"
    print(hdr)
    print("-" * 76)

    for i, frame in enumerate(FRAMES, start=1):
        age     = frame["age"]
        gaze    = frame["eye_gaze"]
        gesture = frame["face_gesture"]
        body    = frame["body_language"]
        spin    = frame["spinning_type"]

        curr = {"eye_gaze": gaze, "face_gesture": gesture,
                "body_language": body, "spinning_type": spin}

        reward_sym = "?"
        if pending:
            reward = model.compute_transition_reward(pending["signals"], curr)
            model.apply_reward(pending["bucket"], pending["response"], reward, curr)
            reward_sym = "✓" if reward > 0 else "✗"
            if model.adaptation_count % 5 == 0:
                model.retrain_from_rewards()

        stim_phase = stim_clf.update(body)
        pred       = model.predict(age, gaze, gesture, body, spin, stim_phase=stim_phase)
        score      = predictor.compute_score(gaze, gesture, body, spin)
        trend      = predictor.trend()

        pending = {
            "bucket":   (pred["age_group"], pred["context"]),
            "response": pred["text"],
            "signals":  curr,
        }

        text_preview = pred["text"][:65].replace("\n", " ")
        print(f"  {i:<2} {score:>5}  {trend:<9} {stim_phase:<16} {reward_sym:<7} {str(pred['explore']):<8} {text_preview}")

        # Validate stim phase detection
        if i in EXPECTED_STIM and stim_phase != EXPECTED_STIM[i]:
            errors.append(f"  Frame {i}: expected stim_phase={EXPECTED_STIM[i]!r}, got {stim_phase!r}")

        # Validate score range
        if not (0 <= score <= 100):
            errors.append(f"  Frame {i}: engagement_score={score} out of 0-100 range")

    print()
    print(f"  Final adaptation count : {model.adaptation_count}")
    print(f"  Reward table buckets   : {len(model.reward_table)}")
    print(f"  Engagement window      : {list(predictor.window)}")

    if errors:
        print()
        print("FAILURES:")
        for e in errors:
            print(e)
        print("\nSelf-test FAILED")
        return 1

    print()
    print("Self-test PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

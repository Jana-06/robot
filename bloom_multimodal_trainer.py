#!/usr/bin/env python3
"""
BLOOM Multimodal Trainer & Inference
====================================
"""
import argparse
import asyncio
import json
import logging
import os
import random
import socket
import subprocess
import sys
import time
from collections import Counter, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("bloom.inference")


def setup_logging() -> None:
    if LOGGER.handlers:
        return
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler("inference.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    LOGGER.addHandler(fh)
    LOGGER.addHandler(sh)


def port_in_use(port: int, host: str = "0.0.0.0") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return False
        except OSError:
            return True


def kill_listeners_on_port(port: int) -> bool:
    try:
        import psutil
    except Exception:
        return False

    killed = False
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            for conn in proc.net_connections(kind="inet"):
                if conn.laddr and conn.laddr.port == port and conn.status == psutil.CONN_LISTEN:
                    LOGGER.warning("Killing old listener pid=%s on port %s", proc.pid, port)
                    proc.kill()
                    killed = True
                    break
        except Exception:
            continue
    return killed


def resolve_server_port(requested_port: int, max_fallbacks: int = 20) -> int:
    if not port_in_use(requested_port):
        return requested_port
    LOGGER.warning("Port %s is busy. Attempting cleanup/fallback.", requested_port)
    if kill_listeners_on_port(requested_port):
        time.sleep(0.4)
        if not port_in_use(requested_port):
            LOGGER.info("Successfully reclaimed requested port %s.", requested_port)
            return requested_port

    for i in range(1, max_fallbacks + 1):
        candidate = requested_port + i
        if not port_in_use(candidate):
            LOGGER.warning("Using fallback port %s instead of %s.", candidate, requested_port)
            return candidate
    raise RuntimeError(f"No available port from {requested_port} to {requested_port + max_fallbacks}")


def persist_runtime_port(port: int) -> None:
    try:
        Path("bloom_multimodal_runtime.json").write_text(
            json.dumps({"port": int(port), "updated_at": int(time.time())}, indent=2),
            encoding="utf-8",
        )
    except Exception as ex:
        LOGGER.warning("Could not persist runtime port: %s", ex)


class DatasetLoader:
    def __init__(self, dataset_root: str):
        self.root = Path(dataset_root)
        self.training_data = self.root / "TrainingDataset"
        self.archive_data = self.root / "archive"

    def load_gaze_data(self) -> Dict[str, Dict]:
        LOGGER.info("Loading eye-tracking data...")
        gaze_data = {}
        asd_fix_dir = self.training_data / "TrainingData" / "ASD"
        td_fix_dir = self.training_data / "TrainingData" / "TD"

        for img_id in range(1, 301):
            gaze_data[img_id] = {"asd": None, "td": None}
            asd_file = asd_fix_dir / f"{img_id}.txt"
            if asd_file.exists():
                gaze_data[img_id]["asd"] = self._parse_fixation_file(asd_file)

            td_file = td_fix_dir / f"{img_id}.txt"
            if td_file.exists():
                gaze_data[img_id]["td"] = self._parse_fixation_file(td_file)

        return gaze_data

    def _parse_fixation_file(self, fpath: Path) -> Optional[Dict]:
        try:
            pts = []
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            pts.append((float(parts[0]), float(parts[1])))
                        except Exception:
                            continue
            return {"points": pts, "count": len(pts)}
        except Exception:
            return None

    def load_screening_data(self) -> pd.DataFrame:
        LOGGER.info("Loading screening data...")
        candidates = [
            self.archive_data / "train.csv",
            self.root / "archive" / "train.csv",
        ]

        p = self.root
        for _ in range(4):
            candidates.append(p / "archive" / "train.csv")
            if p.parent == p:
                break
            p = p.parent

        for cand in candidates:
            try:
                if cand.exists():
                    df = pd.read_csv(cand)
                    LOGGER.info("Screening data loaded from %s", cand)
                    return df
            except Exception as ex:
                LOGGER.warning("Failed reading %s: %s", cand, ex)

        LOGGER.warning("archive/train.csv not found, continuing with empty screening dataframe")
        return pd.DataFrame()

    def load_all(self) -> Tuple[Dict, pd.DataFrame]:
        return self.load_gaze_data(), self.load_screening_data()


class SyntheticDataGenerator:
    def __init__(self, gaze_data: Dict, screening_df: pd.DataFrame):
        self.gaze_data = gaze_data
        self.screening_df = screening_df
        self.response_templates = self._load_response_templates()

    def _load_response_templates(self) -> Dict[str, List[str]]:
        return {
            "5-8_calm": [
                "That sounds fun! Tell me more!",
                "I love that you're thinking about this!",
                "Your idea is so cool!",
                "I'm listening! What happens next?",
                "That's really interesting!",
            ],
            "5-8_stim": [
                "Your stimming helps you feel good! Keep doing what helps your body.",
                "That's a wonderful way to help your body stay regulated!",
                "Your movements are part of how you're awesome!",
                "I see you taking care of yourself! That's great!",
            ],
            "5-8_upset": [
                "That sounds really hard. I'm here with you.",
                "Big feelings are okay. Let's breathe together.",
                "You're safe. What do you need right now?",
                "I hear you. You're not alone in this.",
            ],
            "9-12_calm": [
                "That's a really thoughtful thing to share.",
                "I'm curious about how you think about that!",
                "You've got an interesting perspective.",
                "Keep going - I want to hear your thoughts!",
                "That's cool! What made you think of that?",
            ],
            "9-12_stim": [
                "Your stimming is a healthy way to regulate. Keep doing what works for you!",
                "The way your body moves is part of who you are.",
                "Stimming helps many brilliant people like you stay balanced.",
                "That's your body's way of taking care of itself - respect that!",
            ],
            "9-12_upset": [
                "I can see this is hard for you. That's okay.",
                "Let's take a moment. You're doing the right thing by talking.",
                "Your feelings matter. I'm listening.",
                "Tough days happen. You're handling it.",
            ],
        }

    def generate_training_samples(self, num_samples: int = 1500) -> List[Dict]:
        LOGGER.info("Generating %s synthetic samples...", num_samples)
        samples: List[Dict] = []

        for _ in range(num_samples):
            age = int(np.random.choice([6, 7, 8, 9, 10, 11]))
            age_group = "5-8" if age <= 8 else "9-12"
            is_asd = bool(np.random.choice([True, False], p=[0.5, 0.5]))
            eye_gaze = str(np.random.choice(["center", "left", "right", "down", "scattered"]))
            body_language = str(np.random.choice(["calm", "active", "agitated", "spinning"]))
            face_gesture = str(np.random.choice(["happy", "neutral", "focused", "uncertain"]))
            spinning_type = "spinning" if body_language == "spinning" else "none"

            context = "calm"
            if body_language in ("agitated", "spinning"):
                context = "stim"
            if face_gesture == "uncertain" and np.random.random() > 0.5:
                context = "upset"

            response = str(np.random.choice(self.response_templates.get(f"{age_group}_{context}", self.response_templates["5-8_calm"])))
            samples.append({
                "age": age,
                "age_group": age_group,
                "is_asd": is_asd,
                "eye_gaze": eye_gaze,
                "face_gesture": face_gesture,
                "body_language": body_language,
                "spinning_type": spinning_type,
                "context": context,
                "response": response,
            })
        return samples


class EngagementPredictor:
    def __init__(self, window_size: int = 10):
        self.window = deque(maxlen=window_size)

    def _score_eye(self, eye_gaze: str) -> float:
        return {
            "center": 1.0,
            "left": 0.7,
            "right": 0.7,
            "down": 0.35,
            "scattered": 0.25,
            "unknown": 0.5,
        }.get(eye_gaze, 0.5)

    def _score_face(self, face_gesture: str) -> float:
        return {
            "happy": 1.0,
            "focused": 0.9,
            "neutral": 0.65,
            "uncertain": 0.35,
            "sad": 0.2,
            "afraid": 0.2,
            "unknown": 0.5,
        }.get(face_gesture, 0.5)

    def _score_body(self, body_language: str) -> float:
        return {
            "calm": 1.0,
            "active": 0.75,
            "agitated": 0.3,
            "spinning": 0.55,
            "unknown": 0.5,
        }.get(body_language, 0.5)

    def _score_spin(self, spinning_type: str, body_language: str) -> float:
        if spinning_type == "none":
            return 0.65
        if body_language == "agitated":
            return 0.45
        return 0.6

    def compute_score(self, eye_gaze: str, face_gesture: str, body_language: str, spinning_type: str) -> int:
        score = (
            self._score_eye(eye_gaze) * 0.30
            + self._score_face(face_gesture) * 0.35
            + self._score_body(body_language) * 0.25
            + self._score_spin(spinning_type, body_language) * 0.10
        ) * 100.0
        value = int(round(max(0, min(100, score))))
        self.window.append(value)
        return value

    def trend(self) -> str:
        if len(self.window) < 3:
            return "stable"
        arr = list(self.window)
        recent = arr[-3:]
        delta = recent[-1] - recent[0]
        avg = sum(recent) / 3.0
        if avg < 30 and delta < -8:
            return "crisis"
        if delta > 7:
            return "rising"
        if delta < -7:
            return "falling"
        return "stable"


class StimPatternClassifier:
    def __init__(self, window_size: int = 5):
        self.body_window = deque(maxlen=window_size)
        self.prev_phase = "none"

    def update(self, body_language: str) -> str:
        self.body_window.append(body_language)
        win = list(self.body_window)
        spinning_count = sum(1 for x in win if x == "spinning")

        if len(win) >= 2 and win[-2] != "spinning" and win[-1] == "spinning":
            self.prev_phase = "onset_stim"
            return self.prev_phase
        if spinning_count >= 3 and win[-1] == "spinning":
            self.prev_phase = "sustained_stim"
            return self.prev_phase
        if len(win) >= 2 and win[-2] == "spinning" and win[-1] != "spinning":
            self.prev_phase = "post_stim"
            return self.prev_phase

        if self.prev_phase in ("onset_stim", "sustained_stim") and win[-1] == "spinning":
            return "sustained_stim"
        return "none"


class MultimodalResponseModel:
    def __init__(self, training_samples: Optional[List[Dict]] = None):
        self.training_samples = training_samples or []
        self.response_index = self._build_response_index()
        self.reward_table: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.reward_history: List[Dict] = []
        self.adaptation_count = 0
        self.epsilon = 0.15

    def _build_response_index(self) -> Dict[Tuple[str, str], List[str]]:
        idx: Dict[Tuple[str, str], List[str]] = {}
        for sample in self.training_samples:
            key = (sample.get("age_group", "5-8"), sample.get("context", "calm"))
            idx.setdefault(key, []).append(sample.get("response", ""))
        return idx

    @staticmethod
    def _context(face_gesture: str, body_language: str) -> str:
        if body_language in ("spinning", "agitated"):
            return "stim"
        if face_gesture in ("uncertain", "sad", "afraid"):
            return "upset"
        return "calm"

    @staticmethod
    def _bucket(age: int, context: str) -> Tuple[str, str]:
        return ("5-8" if int(age) <= 8 else "9-12", context)

    def _reward_entry(self, bucket: Tuple[str, str], response: str) -> Dict[str, float]:
        bkey = f"{bucket[0]}::{bucket[1]}"
        btable = self.reward_table.setdefault(bkey, {})
        entry = btable.setdefault(response, {"sum": 0.0, "count": 0.0})
        return entry

    def _response_score(self, bucket: Tuple[str, str], response: str) -> float:
        entry = self._reward_entry(bucket, response)
        if entry["count"] <= 0:
            return 0.0
        return entry["sum"] / entry["count"]

    def select_response(self, age: int, context: str) -> Tuple[str, Tuple[str, str], bool, float]:
        bucket = self._bucket(age, context)
        choices = self.response_index.get(bucket) or []
        if not choices:
            return (
                "That sounds really interesting! Tell me more - I want to hear everything you want to share.",
                bucket,
                False,
                0.0,
            )

        explore = random.random() < self.epsilon
        if explore:
            chosen = random.choice(choices)
        else:
            chosen = max(choices, key=lambda r: self._response_score(bucket, r))
        return chosen, bucket, explore, self._response_score(bucket, chosen)

    def predict(self, age: int, eye_gaze: str, face_gesture: str, body_language: str, spinning_type: str, stim_phase: str) -> Dict:
        context = self._context(face_gesture, body_language)
        response, bucket, explore, score = self.select_response(age, context)

        if stim_phase == "onset_stim":
            response = "I see you're moving - that's totally okay, keep going!"
        elif stim_phase == "sustained_stim":
            response = "I see your body finding its rhythm. You can keep moving in the way that feels right for you."
        elif stim_phase == "post_stim":
            response = "That looked like it felt good. How are you feeling?"

        return {
            "text": response,
            "context": context,
            "age_group": bucket[0],
            "explore": explore,
            "score": round(score, 3),
        }

    def compute_transition_reward(self, prev_signals: Dict, curr_signals: Dict) -> float:
        reward = -0.5

        face_pairs = {
            ("uncertain", "happy"),
            ("uncertain", "neutral"),
            ("sad", "neutral"),
            ("sad", "happy"),
        }
        body_pairs = {
            ("agitated", "calm"),
            ("spinning", "active"),
            ("spinning", "calm"),
        }
        gaze_pairs = {
            ("scattered", "center"),
            ("down", "center"),
        }

        if (prev_signals.get("face_gesture"), curr_signals.get("face_gesture")) in face_pairs:
            reward += 1.0
        if (prev_signals.get("body_language"), curr_signals.get("body_language")) in body_pairs:
            reward += 1.0
        if (prev_signals.get("eye_gaze"), curr_signals.get("eye_gaze")) in gaze_pairs:
            reward += 0.5
        return round(reward, 2)

    def apply_reward(self, bucket: Tuple[str, str], response: str, reward: float, signals: Dict) -> Dict:
        entry = self._reward_entry(bucket, response)
        entry["sum"] += reward
        entry["count"] += 1.0
        hist = {
            "ts": int(time.time()),
            "bucket": f"{bucket[0]}::{bucket[1]}",
            "response": response,
            "reward": reward,
            "signals": signals,
        }
        self.reward_history.append(hist)
        self.adaptation_count += 1
        return hist

    def retrain_from_rewards(self) -> None:
        for bucket_key, responses in self.reward_table.items():
            age_group, context = bucket_key.split("::", 1)
            key = (age_group, context)
            if key not in self.response_index:
                continue
            ranked = sorted(
                self.response_index[key],
                key=lambda r: ((responses.get(r, {}).get("sum", 0.0) / max(1.0, responses.get(r, {}).get("count", 0.0))),),
                reverse=True,
            )
            self.response_index[key] = ranked

    def save(self, path: str) -> None:
        state = {
            "training_samples": self.training_samples,
            "response_index": {f"{k[0]}::{k[1]}": v for k, v in self.response_index.items()},
            "reward_table": self.reward_table,
            "reward_history": self.reward_history[-5000:],
            "adaptation_count": self.adaptation_count,
            "epsilon": self.epsilon,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        LOGGER.info("Model saved to %s", path)

    @staticmethod
    def load(path: str) -> "MultimodalResponseModel":
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        model = MultimodalResponseModel(training_samples=state.get("training_samples", []))

        stored_index = state.get("response_index") or {}
        if stored_index:
            model.response_index = {}
            for key, value in stored_index.items():
                if "::" in key:
                    a, c = key.split("::", 1)
                    model.response_index[(a, c)] = value

        model.reward_table = state.get("reward_table", {})
        model.reward_history = state.get("reward_history", [])
        model.adaptation_count = int(state.get("adaptation_count", 0))
        model.epsilon = float(state.get("epsilon", 0.15))
        return model


async def start_inference_server(model: MultimodalResponseModel, model_path: str, port: int = 8766):
    try:
        import websockets
        from websockets.asyncio.server import serve as ws_serve
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "websockets", "-q"], check=False)
        import websockets
        from websockets.asyncio.server import serve as ws_serve

    port = resolve_server_port(port)
    persist_runtime_port(port)
    LOGGER.info("BLOOM inference server listening on ws://localhost:%s", port)

    async def heartbeat(ws):
        while True:
            await asyncio.sleep(30)
            try:
                await ws.send(json.dumps({"type": "heartbeat", "ts": int(time.time())}))
            except Exception:
                return

    async def handler(ws, path=None):
        LOGGER.info("Client connected to inference server")
        predictor = EngagementPredictor(window_size=10)
        stim_classifier = StimPatternClassifier(window_size=5)
        pending_reward = None
        hb_task = asyncio.create_task(heartbeat(ws))

        try:
            async for message in ws:
                try:
                    req = json.loads(message)
                    if req.get("type") != "predict":
                        continue

                    age = int(req.get("age", 8))
                    gaze = str(req.get("eye_gaze", "center"))
                    gesture = str(req.get("face_gesture", "neutral"))
                    body = str(req.get("body_language", "calm"))
                    spin = str(req.get("spinning_type", "none"))

                    curr_signals = {
                        "eye_gaze": gaze,
                        "face_gesture": gesture,
                        "body_language": body,
                        "spinning_type": spin,
                    }

                    last_reward = None
                    reward_symbol = "?"
                    rl_updated = False
                    if pending_reward:
                        reward = model.compute_transition_reward(pending_reward["signals"], curr_signals)
                        h = model.apply_reward(
                            pending_reward["bucket"],
                            pending_reward["response"],
                            reward,
                            curr_signals,
                        )
                        last_reward = reward
                        reward_symbol = "✓" if reward > 0 else "✗"
                        rl_updated = True
                        if model.adaptation_count % 5 == 0:
                            model.retrain_from_rewards()
                        model.save(model_path)
                        LOGGER.info("Reward update: %s", h)

                    stim_phase = stim_classifier.update(body)
                    pred = model.predict(age, gaze, gesture, body, spin, stim_phase=stim_phase)
                    engagement_score = predictor.compute_score(gaze, gesture, body, spin)
                    engagement_trend = predictor.trend()

                    pending_reward = {
                        "bucket": (pred["age_group"], pred["context"]),
                        "response": pred["text"],
                        "signals": curr_signals,
                    }

                    out = {
                        "type": "response",
                        "text": pred["text"],
                        "age": age,
                        "eye_gaze": gaze,
                        "face_gesture": gesture,
                        "body_language": body,
                        "spinning_type": spin,
                        "context": pred["context"],
                        "engagement_score": engagement_score,
                        "engagement_trend": engagement_trend,
                        "stim_phase": stim_phase,
                        "reward_feedback": reward_symbol,
                        "last_reward": last_reward,
                        "rl_selected_explore": pred["explore"],
                        "rl_response_score": pred["score"],
                        "rl_adapted": rl_updated,
                        "adaptation_count": model.adaptation_count,
                    }
                    await ws.send(json.dumps(out))
                except Exception as ex:
                    LOGGER.exception("Inference handler error")
                    await ws.send(json.dumps({"type": "error", "message": str(ex)}))
        except Exception:
            LOGGER.exception("Client loop error")
        finally:
            hb_task.cancel()
            LOGGER.info("Client disconnected from inference server")

    async with ws_serve(handler, "0.0.0.0", port, max_size=10 * 1024 * 1024, ping_interval=30, ping_timeout=30):
        await asyncio.Future()


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="BLOOM Multimodal Trainer")
    parser.add_argument("--dataset-root", default=None, help="Path to dataset root")
    parser.add_argument("--train", action="store_true", help="Generate and train on synthetic data")
    parser.add_argument("--serve", action="store_true", help="Start inference server")
    parser.add_argument("--retrain", action="store_true", help="Re-rank policy from accumulated rewards")
    parser.add_argument("--model-path", default="bloom_multimodal_model.json", help="Model save/load path")
    parser.add_argument("--port", type=int, default=8766, help="Inference server port")
    args = parser.parse_args()

    if args.train:
        if not args.dataset_root:
            parser.error("--dataset-root is required when using --train")
        LOGGER.info("BLOOM training pipeline starting")
        loader = DatasetLoader(args.dataset_root)
        gaze_data, screening_df = loader.load_all()
        LOGGER.info("Loaded %s eye-tracking samples", len(gaze_data))
        LOGGER.info("Loaded %s screening rows", len(screening_df))

        samples = SyntheticDataGenerator(gaze_data, screening_df).generate_training_samples(num_samples=1500)
        model = MultimodalResponseModel(training_samples=samples)
        model.save(args.model_path)
        LOGGER.info("Training complete")
        return

    if args.retrain:
        if not Path(args.model_path).exists():
            LOGGER.error("Model not found at %s", args.model_path)
            sys.exit(1)
        model = MultimodalResponseModel.load(args.model_path)
        model.retrain_from_rewards()
        model.save(args.model_path)
        LOGGER.info("Retrain complete using reward table")
        return

    if args.serve:
        if not Path(args.model_path).exists():
            LOGGER.error("Model not found at %s", args.model_path)
            LOGGER.error("Train first: python bloom_multimodal_trainer.py --train --dataset-root <path>")
            sys.exit(1)
        model = MultimodalResponseModel.load(args.model_path)
        LOGGER.info("Model loaded from %s", args.model_path)
        try:
            asyncio.run(start_inference_server(model, args.model_path, port=args.port))
        except KeyboardInterrupt:
            LOGGER.info("Inference server stopped")
        return

    parser.print_help()


if __name__ == "__main__":
    main()

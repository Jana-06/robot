#!/usr/bin/env python3
import asyncio
import json

import websockets


FRAMES = [
    {"age": 8, "eye_gaze": "scattered", "face_gesture": "uncertain", "body_language": "agitated", "spinning_type": "none"},
    {"age": 8, "eye_gaze": "center", "face_gesture": "happy", "body_language": "calm", "spinning_type": "none"},
    {"age": 8, "eye_gaze": "down", "face_gesture": "neutral", "body_language": "spinning", "spinning_type": "spinning"},
    {"age": 8, "eye_gaze": "center", "face_gesture": "focused", "body_language": "spinning", "spinning_type": "spinning"},
    {"age": 8, "eye_gaze": "center", "face_gesture": "focused", "body_language": "spinning", "spinning_type": "spinning"},
    {"age": 8, "eye_gaze": "center", "face_gesture": "focused", "body_language": "active", "spinning_type": "none"},
    {"age": 9, "eye_gaze": "left", "face_gesture": "uncertain", "body_language": "agitated", "spinning_type": "none"},
    {"age": 9, "eye_gaze": "center", "face_gesture": "neutral", "body_language": "calm", "spinning_type": "none"},
    {"age": 9, "eye_gaze": "center", "face_gesture": "happy", "body_language": "calm", "spinning_type": "none"},
    {"age": 9, "eye_gaze": "scattered", "face_gesture": "sad", "body_language": "agitated", "spinning_type": "none"},
]


async def main() -> None:
    async with websockets.connect("ws://localhost:8766", max_size=10 * 1024 * 1024) as ws:
        for i, frame in enumerate(FRAMES, start=1):
            req = {"type": "predict", **frame}
            await ws.send(json.dumps(req))
            resp = json.loads(await ws.recv())
            text = (resp.get("text") or "")[:68].replace("\n", " ")
            print(
                f"frame={i:02d} score={resp.get('engagement_score')} trend={resp.get('engagement_trend')} "
                f"stim={resp.get('stim_phase')} reward={resp.get('reward_feedback')} "
                f"explore={resp.get('rl_selected_explore')} text={text}"
            )


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
Launcher for the BLOOM demo project.

Starts the multimodal inference server (ws://localhost:8766) and the BLOOM
WebSocket server (ws://localhost:$PORT).  Keeps both alive forever, restarting
any process that exits unexpectedly.

Usage: python maini.py
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
TRAINER = os.path.join(ROOT, "bloom_multimodal_trainer.py")
SERVER  = os.path.join(ROOT, "bloom_server.py")

MM_URL  = "ws://localhost:8766"
BS_PORT = int(os.environ.get("PORT", 8765))
BS_URL  = f"ws://localhost:{BS_PORT}"


def start_process(cmd, name):
    print(f"Starting: {name}", flush=True)
    if os.name == "nt":
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_CONSOLE,
        )
    return subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)


def probe_ws(url, timeout=30.0):
    import asyncio
    try:
        import websockets
    except Exception:
        subprocess.run([PY, "-m", "pip", "install", "websockets", "-q"])
        import websockets

    async def _check():
        try:
            async with websockets.connect(url, max_size=10 * 1024 * 1024):
                return True
        except Exception:
            return False

    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            ok = asyncio.run(_check())
        except Exception:
            ok = False
        if ok:
            return True
        time.sleep(0.6)
    return False


def ensure_stim_model():
    stim_model = os.path.join(ROOT, "bloom_stim_model.json")
    if os.path.exists(stim_model):
        return
    print("Training stim model... (first time only, ~30-60 seconds)", flush=True)
    try:
        result = subprocess.run(
            [PY, TRAINER, "--train-stim", "--stim-model-path", stim_model],
            timeout=120,
            check=False,
        )
        if result.returncode == 0:
            print("Stim model training complete.", flush=True)
        else:
            print("Stim model training exited with errors — continuing without it.", flush=True)
    except subprocess.TimeoutExpired:
        print("Stim model training timed out — continuing without it.", flush=True)
    except Exception as e:
        print(f"Stim model training failed: {e} — continuing without it.", flush=True)


def main():
    ensure_stim_model()

    model_file = os.path.join(ROOT, "bloom_multimodal_model.json")
    if not os.path.exists(model_file):
        print(
            "Warning: bloom_multimodal_model.json not found — "
            "trainer will run in empty-mode.",
            flush=True,
        )

    trainer_cmd = [PY, TRAINER, "--serve", "--model-path", "bloom_multimodal_model.json"]
    server_cmd  = [PY, SERVER,  "--child", "demo"]

    p_trainer = start_process(trainer_cmd, "multimodal trainer  (ws://localhost:8766 internal)")
    p_server  = start_process(server_cmd,  f"bloom server        (ws://localhost:{BS_PORT} public)")

    print("Probing services — this may take up to 30 s on first start…", flush=True)
    mm_ok = probe_ws(MM_URL, timeout=30)
    bs_ok = probe_ws(BS_URL, timeout=30)
    print(f"Multimodal server reachable : {mm_ok}", flush=True)
    print(f"BLOOM server reachable      : {bs_ok}", flush=True)
    print(f"Trainer PID : {p_trainer.pid}", flush=True)
    print(f"Server PID  : {p_server.pid}",  flush=True)
    print("Both servers running. Keeping alive — press Ctrl-C to stop.", flush=True)

    # Keep-alive loop: restart any child that exits unexpectedly.
    while True:
        time.sleep(5)
        if p_trainer.poll() is not None:
            print("Trainer exited unexpectedly; restarting…", flush=True)
            p_trainer = start_process(trainer_cmd, "multimodal trainer (restart)")
        if p_server.poll() is not None:
            print("Server exited unexpectedly; restarting…", flush=True)
            p_server = start_process(server_cmd, "bloom server (restart)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Launcher stopped.", flush=True)

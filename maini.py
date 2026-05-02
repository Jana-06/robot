#!/usr/bin/env python3
"""
Launcher for the BLOOM demo project.

Starts the multimodal inference server and the BLOOM WebSocket server,
waits for them to become reachable, then opens the dashboard HTML.

Usage: python maini.py
"""
import subprocess, sys, time, os, json
from urllib.parse import urlparse

ROOT = os.path.dirname(__file__)
PY = sys.executable
TRAINER = os.path.join(ROOT, "bloom_multimodal_trainer.py")
SERVER = os.path.join(ROOT, "bloom_server.py")
DASH = os.path.join(ROOT, "bloom_dashboard.html")

MM_URL = "ws://localhost:8766"
BS_URL = "ws://localhost:8765"

def start_process(cmd, name):
    print(f"Starting: {name}")
    # On Windows, create a new console so logs are visible separately
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_CONSOLE
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=creationflags, shell=False)

def probe_ws(url, timeout=20.0):
    import asyncio
    try:
        import websockets
    except Exception:
        subprocess.run([PY, "-m", "pip", "install", "websockets", "-q"]) ; import websockets

    async def _check():
        try:
            async with websockets.connect(url, max_size=10*1024*1024):
                return True
        except Exception:
            return False

    t0=time.time()
    while time.time()-t0 < timeout:
        try:
            ok = asyncio.run(_check())
        except Exception:
            ok = False
        if ok: return True
        time.sleep(0.6)
    return False

def main():
    # Ensure model file exists (trainer uses it when serving)
    model_file = os.path.join(ROOT, "bloom_multimodal_model.json")
    if not os.path.exists(model_file):
        print("Warning: bloom_multimodal_model.json not found — trainer will generate or run in empty-mode.")

    trainer_cmd = [PY, TRAINER, "--serve", "--model-path", "bloom_multimodal_model.json"]
    server_cmd = [PY, SERVER, "--child", "demo"]

    p_trainer = start_process(trainer_cmd, "multimodal trainer")
    p_server = start_process(server_cmd, "bloom server")

    print("Probing services (this may take a few seconds)...")
    mm_ok = probe_ws(MM_URL, timeout=30)
    bs_ok = probe_ws(BS_URL, timeout=30)

    print(f"Multimodal server reachable: {mm_ok}")
    print(f"BLOOM server reachable: {bs_ok}")

    if os.path.exists(DASH):
        try:
            import webbrowser
            webbrowser.open('file://' + DASH)
            print(f"Opened dashboard: {DASH}")
        except Exception as e:
            print(f"Could not open dashboard: {e}")
    else:
        print("Dashboard HTML not found; open bloom_dashboard.html manually.")

    print("Launcher finished. Server logs are running in their consoles.")
    print("If you want to stop both servers, kill the PIDs printed below.")
    print("Trainer PID:", p_trainer.pid)
    print("Server PID:", p_server.pid)

if __name__ == '__main__':
    main()

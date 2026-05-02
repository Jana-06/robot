@echo off
REM BLOOM Multimodal Quick-Start for Windows
REM ==========================================

echo.
echo 🌸 BLOOM Multimodal Setup
echo ========================
echo.

REM Step 1: Install dependencies
echo Step 1: Installing Python dependencies...
pip install -r requirements.txt -q
if %errorlevel% neq 0 (
    echo ✗ Failed to install dependencies
    exit /b 1
)
echo ✓ Dependencies installed
echo.

REM Step 2: Train the model
echo Step 2: Training multimodal model (this may take a minute)...
python bloom_multimodal_trainer.py --dataset-root "C:\Users\Janarthan S\Downloads" --train
if %errorlevel% neq 0 (
    echo ✗ Training failed
    exit /b 1
)
echo ✓ Model trained
echo.

REM Step 3: Display next steps
echo ✓ Setup complete!
echo.
echo Next steps:
echo.
echo 1. Start the multimodal inference server (in a new terminal):
echo    python bloom_multimodal_trainer.py --serve
echo.
echo 2. In another terminal, start BLOOM server:
echo    python bloom_server.py --child arjun
echo.
echo 3. Open the dashboard:
echo    bloom_dashboard.html
echo.
echo 4. (Optional) Test the bridge:
echo    python bloom_multimodal_bridge.py
echo.

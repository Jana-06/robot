# BLOOM Multimodal Quick-Start for PowerShell
# ==========================================

Write-Host ""
Write-Host "🌸 BLOOM Multimodal Setup" -ForegroundColor Cyan
Write-Host "========================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Install dependencies
Write-Host "Step 1: Installing Python dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt -q
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Failed to install dependencies" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Dependencies installed" -ForegroundColor Green
Write-Host ""

# Step 2: Train the model
Write-Host "Step 2: Training multimodal model (this may take a minute)..." -ForegroundColor Yellow
python bloom_multimodal_trainer.py --dataset-root "C:\Users\Janarthan S\Downloads" --train
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Training failed" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Model trained" -ForegroundColor Green
Write-Host ""

# Step 3: Display next steps
Write-Host "✓ Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Start the multimodal inference server (in a new PowerShell terminal):" -ForegroundColor White
Write-Host "   python bloom_multimodal_trainer.py --serve" -ForegroundColor Cyan
Write-Host ""
Write-Host "2. In another terminal, start BLOOM server:" -ForegroundColor White
Write-Host "   python bloom_server.py --child arjun" -ForegroundColor Cyan
Write-Host ""
Write-Host "3. Open the dashboard:" -ForegroundColor White
Write-Host "   bloom_dashboard.html" -ForegroundColor Cyan
Write-Host ""
Write-Host "4. (Optional) Test the bridge:" -ForegroundColor White
Write-Host "   python bloom_multimodal_bridge.py" -ForegroundColor Cyan
Write-Host ""

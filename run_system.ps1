Write-Host " Starting Clinical NER Automation System..." -ForegroundColor Cyan
# Activate environment
.\venv\Scripts\Activate.ps1
# Start the API
uvicorn src.main:app --reload
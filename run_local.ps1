<#
.SYNOPSIS
Script khoi dong Chatbot Phap Luat VN - GROQ API VERSION (Sieu toc)
#>

Write-Host "==========================================" -ForegroundColor Green
Write-Host " KHOI DONG LOCAL CHATBOT VBPL - GROQ API" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green

Write-Host "`n[1/4] Khoi tao va kich hoat venv..." -ForegroundColor Cyan
if (-Not (Test-Path "venv")) {
    python -m venv venv
}
. .\venv\Scripts\Activate.ps1

Write-Host "`n[2/4] Kiem tra thu vien Python..." -ForegroundColor Cyan
if (-Not (Get-Command streamlit -ErrorAction SilentlyContinue)) {
    Write-Host "Dang cai dat cac thu vien (Chi 1 lan duy nhat)..."
    pip install -r requirements.txt --quiet
}

Write-Host "`n[3/4] Thiet lap bien moi truong (.env) cho GROQ..." -ForegroundColor Cyan
if (-Not (Test-Path ".env")) {
    $apiKey = Read-Host "Ban chua co file .env. Vui long nhap GROQ_API_KEY cua ban vao day (Bat buoc)"
    if ([string]::IsNullOrWhiteSpace($apiKey)) {
        $apiKey = "YOUR_GROQ_API_KEY"
    }
    $geminiKey = Read-Host "Nhap GEMINI_API_KEY neu muon test Gemini 3 Flash (co the bo trong)"
    
    $envContent = @"
QDRANT_URL=./local_qdrant_db
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_PROVIDER=groq
LLM_API_KEY=$apiKey
LLM_CHAT_MODEL=llama3-8b-8192
GEMINI_API_KEY=$geminiKey
GEMINI_CHAT_MODEL=gemini-3-flash-preview
"@
    Set-Content -Path ".env" -Value $envContent
    Write-Host "Da luu cau hinh vao file .env. De sua KEY sau nay, chi can sua file nay!" -ForegroundColor Green
} else {
    Write-Host "File .env da ton tai. Neu gap loi 'Invalid API Key', hay mo file .env de sua lai LLM_API_KEY nhe." -ForegroundColor Yellow
}

Write-Host "`n[4/4] Khoi dong Giao dien Chatbot..." -ForegroundColor Cyan
streamlit run frontend\app.py

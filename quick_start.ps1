<#
.SYNOPSIS
Script khởi động Backend & Frontend của hệ thống Legal-RAG (Local).
Dành cho test local với SQLite và Redis container.
#>

Write-Host "==========================================" -ForegroundColor Green
Write-Host " KHOI DONG LEGAL RAG - BACKEND & FRONTEND" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green

Push-Location $PSScriptRoot

# 1. Start Redis & Qdrant in Docker
Write-Host "`n[1/6] Khoi dong DB Services (Redis & Qdrant) via Docker..." -ForegroundColor Cyan
if (-Not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "LỖI: Khong tim thay Docker CLI. Vui long cai dat Docker Desktop!" -ForegroundColor Red
    Exit
}

$runningContainers = docker ps --filter "status=running" --format "{{.Names}}"
if ($runningContainers -notcontains "legal-rag-redis" -or $runningContainers -notcontains "legal-rag-qdrant") {
    Write-Host "Phat hien thieu dich vu DB. Dang khoi chay qua Docker Compose..." -ForegroundColor Yellow
    docker-compose -f docker-compose.yml up -d
}
else {
    Write-Host "Cac dich vu Redis & Qdrant da san sang (Up & Running)." -ForegroundColor Green
}

# 2. Setup Python Venv
Write-Host "`n[2/6] Kiem tra va khoi tao Moi truong Python (venv)..." -ForegroundColor Cyan
if (-Not (Test-Path "venv")) {
    Write-Host "Dang tao Virtual Environment (venv) moi cho thiet bi..."
    python -m venv venv
}
. .\venv\Scripts\Activate.ps1

# 3. Install Requirements
Write-Host "`n[3/6] Cai dat/Cap nhat thu vien LangGraph & RAG Core (pip install)..." -ForegroundColor Cyan
pip install -r requirements.txt --quiet --no-warn-script-location

# 4. Check/Create .env & data folder
Write-Host "`n[4/6] Kiem tra file cau hinh (.env) va thu muc data (SQLite)..." -ForegroundColor Cyan
if (-Not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" -Destination ".env"
        Write-Host "Da tao file .env tu .env.example. Vui long KIEM TRA LAI thong tin Cloud Qdrant ban nhe!" -ForegroundColor Yellow
    }
    else {
        Write-Host "Warning: Khong tim thay .env hay .env.example!" -ForegroundColor Red
    }
}
else {
    Write-Host "File .env da ton tai." -ForegroundColor Green
}
if (-Not (Test-Path "backend/data")) {
    New-Item -ItemType Directory -Path "backend/data" -Force | Out-Null
    Write-Host "Da tao thu muc backend/data de luu tru SQLite." -ForegroundColor Green
}
if (-Not (Test-Path "backend/tmp_uploads")) {
    New-Item -ItemType Directory -Path "backend/tmp_uploads" -Force | Out-Null
    Write-Host "Da tao thu muc backend/tmp_uploads de luu tru file tam." -ForegroundColor Green
}

# 5. Dọn dẹp cổng cũ — tắt hết các tiến trình Next.js/FastAPI/Embedding còn sót
Write-Host "`n[5/6] Don dep cac cong cu (3000-3002, 8000, 8001)..." -ForegroundColor Cyan
$portsToKill = @(3000, 3001, 3002, 8000, 8001)
foreach ($port in $portsToKill) {
    $pidsToRemove = (Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue).OwningProcess | Sort-Object -Unique
    foreach ($p in $pidsToRemove) {
        if ($p -and $p -ne 0 -and $p -ne $PID) {
            $proc = Get-Process -Id $p -ErrorAction SilentlyContinue
            if ($proc) {
                Write-Host "  Kill PID $p ($($proc.ProcessName)) dang chiem port $port" -ForegroundColor Yellow
                Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
            }
        }
    }
}
Start-Sleep -Seconds 2

# 6. Start Services
Write-Host "`n[6/6] Khoi dong cac dich vu (Embedding, API, Celery, Next.js)..." -ForegroundColor Cyan

# Start Embedding Server
Start-Process powershell -WorkingDirectory "$PSScriptRoot" -ArgumentList "-NoExit", "-WindowStyle", "Normal", "-Title", "'Embedding Server'", "-Command", ". .\venv\Scripts\Activate.ps1; echo 'Starting Embedding Server (Port 8001)...'; python -m backend.retrieval.server"

# Start FastAPI
Start-Process powershell -WorkingDirectory "$PSScriptRoot" -ArgumentList "-NoExit", "-WindowStyle", "Normal", "-Title", "'FastAPI Backend'", "-Command", ". .\venv\Scripts\Activate.ps1; echo 'Starting FastAPI Backend (Port 8000)...'; uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload"

# Start Celery Worker
Start-Process powershell -WorkingDirectory "$PSScriptRoot" -ArgumentList "-NoExit", "-WindowStyle", "Normal", "-Title", "'Celery Worker (Task Queue)'", "-Command", ". .\venv\Scripts\Activate.ps1; echo 'Starting Celery Background Worker...'; celery -A backend.workers.celery_app worker --loglevel=info -P solo"

Write-Host "Dang cho cac backend services khoi dong an toan..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Start Next.js Frontend — Cố định cổng 3000
Start-Process powershell -WorkingDirectory "$PSScriptRoot/frontend" -ArgumentList "-NoExit", "-WindowStyle", "Normal", "-Title", "'Next.js Frontend'", "-Command", "`$env:PORT=3000; npm run dev -- --port 3000"

Write-Host "`nHoan tat! He thong Legal-RAG (Local) dang duoc khoi dong tren nhieu cuaso." -ForegroundColor Green
Write-Host "Frontend: http://localhost:3000  |  API: http://localhost:8000" -ForegroundColor Cyan
Write-Host "Vui long doi Embedding Server tai model (bge-m3) neu day la lan dau chay tren may!" -ForegroundColor Yellow
Write-Host "`nLuu y: He thong hien da ho tro Che do Tai len Tam thoi (Staged Ingestion) giup truy van nhanh trong RAM!" -ForegroundColor Magenta


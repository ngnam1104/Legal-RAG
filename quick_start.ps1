<#
.SYNOPSIS
Script khoi dong Backend & Frontend cua he thong Legal-RAG (Local).
Dành cho test local với SQLite và Redis container.
#>

Write-Host "==========================================" -ForegroundColor Green
Write-Host " KHOI DONG LEGAL RAG - BACKEND & FRONTEND" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green

Push-Location $PSScriptRoot

# 1. Start Redis & Qdrant in Docker
Write-Host "`n[1/6] Khoi dong DB Services (Redis & Qdrant) via Docker..." -ForegroundColor Cyan
if (-Not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "LOI: Khong tim thay Docker CLI. Vui long cai dat Docker Desktop!" -ForegroundColor Red
    Exit
}

$runningContainers = docker ps --filter "status=running" --format "{{.Names}}"
if ($runningContainers -notcontains "legal-rag-redis" -or $runningContainers -notcontains "legal-rag-qdrant" -or $runningContainers -notcontains "legal-rag-neo4j") {
    Write-Host "Phat hien thieu dich vu DB (Redis / Qdrant / Neo4j). Dang khoi chay qua Docker Compose..." -ForegroundColor Yellow
    docker-compose -f docker-compose.yml up -d
}
else {
    Write-Host "Cac dich vu Redis, Qdrant & Neo4j da san sang (Up & Running)." -ForegroundColor Green
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
        Write-Host "Da tao file .env tu .env.example. Vui long KIEM TRA LAI thong tin DB & Neo4j ban nhe!" -ForegroundColor Yellow
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

# 5. Don dep cong cu — tat het cac tien trinh Next.js/FastAPI/Embedding con sot
Write-Host "`n[5/6] Don dep cac cong cu (3000-3001, 8000, 8001)..." -ForegroundColor Cyan
$portsToKill = @(3000, 3001, 8000, 8001)
foreach ($port in $portsToKill) {
    try {
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
    } catch {}
}
Start-Sleep -Seconds 2

# 6. Start Services
Write-Host "`n[6/6] Khoi dong cac dich vu (FastAPI & Next.js)..." -ForegroundColor Cyan

# Start FastAPI Backend
# Su dung mot chuoi ArgumentList duy nhat de tranh loi parser
$fastapiArgs = "-NoExit -Command `"`$Host.UI.RawUI.WindowTitle='FastAPI Backend'; . .\venv\Scripts\Activate.ps1; echo 'Starting FastAPI...'; uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir=backend`""
Start-Process powershell -WorkingDirectory "$PSScriptRoot" -ArgumentList $fastapiArgs

Write-Host "VUI LONG DOI: Backend dang thuc hien WARMUP (Nap model vao RAM)..." -ForegroundColor Yellow
Write-Host "Thoi gian cho du kien: 20-30 giay de dam bao truy van 'Zero-Wait' ngay tu cau dau tien." -ForegroundColor Gray
Start-Sleep -Seconds 20

# Start Next.js Frontend — Co dinh cong 3000
$frontendArgs = "-NoExit -Command `"`$Host.UI.RawUI.WindowTitle='Next.js Frontend'; `$env:PORT=3000; npm run dev -- --port 3000`""
Start-Process powershell -WorkingDirectory "$PSScriptRoot/frontend" -ArgumentList $frontendArgs

Write-Host "`n============================================================" -ForegroundColor Green
Write-Host " HOAN TAT! HE THONG LEGAL-RAG (LOCAL) DA SAN SANG" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  - Frontend: http://localhost:3000" -ForegroundColor Cyan
Write-Host "  - Backend API: http://localhost:8000" -ForegroundColor Cyan
Write-Host "  - Neo4j Graph DB: http://localhost:7474" -ForegroundColor Cyan
Write-Host "------------------------------------------------------------" -ForegroundColor DarkGray
Write-Host "TINH NANG MOI DA DUOC CAP NHAT (DUAL RAG UPDATE):" -ForegroundColor Magenta
Write-Host "  [+] Architecture: Dual Store DB (Neo4j Graph & Qdrant Vector) xu ly triet de 0 hits." -ForegroundColor White
Write-Host "  [+] Intent Router: Auto-Routing thong minh (QA, Search, Conflict)." -ForegroundColor White
Write-Host "  [+] Preamble Inheritance: Bao ton va di truyen can cu phap ly o header." -ForegroundColor White
Write-Host "  [+] Lex Posterior Safe: Loc loai bo du lieu het hieu luc chinh xac." -ForegroundColor White
Write-Host "------------------------------------------------------------" -ForegroundColor DarkGray
Write-Host "Vui long kiem tra cac cua so Terminal de theo doi log chi tiet." -ForegroundColor Yellow
Write-Host "Chuc ban co trai nghiem tra cuu phap luat tuyet voi!" -ForegroundColor Green

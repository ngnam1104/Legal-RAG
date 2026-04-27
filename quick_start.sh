#!/bin/bash

# ==============================================================================
# Script khoi dong Backend & Frontend cua he thong Legal-RAG (Ubuntu/Server).
# Dành cho test local với SQLite và Docker container.
# ==============================================================================

# Define colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}==========================================${NC}"
echo -e "${GREEN} KHOI DONG LEGAL RAG - BACKEND & FRONTEND${NC}"
echo -e "${GREEN}==========================================${NC}"

# Chuyển về đúng thư mục chứa script
cd "$(dirname "$0")" || exit

# 1. Start Docker containers
echo -e "\n${CYAN}[1/6] Khoi dong DB Services (Redis, Qdrant, Neo4j) via Docker...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}LOI: Khong tim thay Docker CLI. Vui long cai dat Docker!${NC}"
    exit 1
fi

# Chạy Docker Compose
echo -e "${YELLOW}Kiem tra/Khoi dong cac container (Redis/Qdrant/Neo4j)...${NC}"
if docker compose version &> /dev/null; then
    docker compose up -d
else
    docker-compose up -d
fi

# 2. Setup Venv
echo -e "\n${CYAN}[2/6] Kiem tra va khoi tao Moi truong Python (venv)...${NC}"
if [ ! -d "venv" ]; then
    echo "Dang tao Virtual Environment (venv)..."
    python3 -m venv venv
fi
source venv/bin/activate

# 3. Install requirements
echo -e "\n${CYAN}[3/6] Cai dat/Cap nhat thu vien (pip install)...${NC}"
pip install -r requirements.txt --quiet

# 4. Check files
echo -e "\n${CYAN}[4/6] Kiem tra file cau hinh (.env) va thu muc...${NC}"
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "${YELLOW}Da tao .env tu .env.example. Vui long kiem tra lai config IP máy chủ/Port DB!${NC}"
    else
        echo -e "${RED}Warning: Khong tim thay .env hay .env.example!${NC}"
    fi
else
    echo -e "${GREEN}File .env da ton tai.${NC}"
fi

mkdir -p backend/data backend/tmp_uploads
echo -e "${GREEN}Thu muc backend/data va backend/tmp_uploads da san sang.${NC}"

# 5. Kill old processes
echo -e "\n${CYAN}[5/6] Don dep cac tien trinh cu (port 3000, 3001, 8000, 8001)...${NC}"
for PORT in 3000 3001 8000 8001; do
    # Tìm tiến trình đang giữ port và kill nó
    PID=$(lsof -t -i:$PORT 2>/dev/null)
    if [ -n "$PID" ]; then
        echo -e "${YELLOW}  Giet tien trinh PID $PID dang chiem port $PORT${NC}"
        kill -9 $PID 2>/dev/null
    fi
done
sleep 2

# 6. Start Services
echo -e "\n${CYAN}[6/6] Khoi dong cac dich vu...${NC}"

# Start Backend
echo "Starting FastAPI Backend..."
nohup uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir=backend > backend_log.txt 2>&1 &

echo -e "${YELLOW}VUI LONG DOI: Backend dang thuc hien WARMUP (nap model, cache) trong vong 20s...${NC}"
sleep 20

# Start Frontend
echo "Starting Next.js Frontend..."
cd frontend || exit
export PORT=3000
nohup npm run dev -- --port 3000 > frontend_log.txt 2>&1 &
cd ..

echo -e "\n${GREEN}============================================================${NC}"
echo -e "${GREEN} HOAN TAT! HE THONG LEGAL-RAG DA SAN SANG (RUNNING IN BACKGROUND)${NC}"
echo -e "${GREEN}============================================================${NC}"
echo -e "${CYAN}  - Frontend: http://IP_MAY_CHU:3000${NC}"
echo -e "${CYAN}  - Backend API: http://IP_MAY_CHU:8000${NC}"
echo -e "${CYAN}  - Neo4j Graph DB: http://IP_MAY_CHU:7474${NC}"
echo -e "${YELLOW}------------------------------------------------------------${NC}"
echo -e "Cac tien trinh da duoc day xuong chay ngam, ban co the an tam tat cua so Terminal."
echo -e "De xem tien trinh dang chay (log):"
echo -e "  - Logs Backend:  ${YELLOW}tail -f backend_log.txt${NC}"
echo -e "  - Logs Frontend: ${YELLOW}tail -f frontend/frontend_log.txt${NC}"
echo -e "${GREEN}Chuc ban co trai nghiem tra cuu phap luat tuyet voi!${NC}"

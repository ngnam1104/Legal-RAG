#!/bin/bash

# ==============================================================================
# Legal-RAG: Script khoi dong Backend & Frontend hoac chay test ingestion
# Su dung:
#   ./quick_start.sh              → Khoi dong Backend + Frontend (mac dinh)
#   ./quick_start.sh --mode=test-ingest  → Chay pipeline test 500 VB y te
# ==============================================================================

GREEN='\033[0;32m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

MODE="start"
for arg in "$@"; do
    case $arg in --mode=*) MODE="${arg#*=}" ;; esac
done

cd "$(dirname "$0")" || exit

# --- 1. Docker containers ---
echo -e "\n${CYAN}[1] Khoi dong DB Services (Redis, Qdrant, Neo4j) via Docker...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}LOI: Khong tim thay Docker CLI!${NC}"; exit 1
fi
if docker compose version &> /dev/null; then
    docker compose up -d
else
    docker-compose up -d
fi

# --- 2. Venv ---
echo -e "\n${CYAN}[2] Kiem tra Python venv...${NC}"
if [ ! -d "venv" ]; then
    echo "Dang tao venv..."; python3 -m venv venv
fi
source venv/bin/activate

# --- 3. Dependencies ---
echo -e "\n${CYAN}[3] Cai dat thu vien...${NC}"
pip install -r requirements.txt --quiet

# --- 4. .env check ---
echo -e "\n${CYAN}[4] Kiem tra .env...${NC}"
if [ ! -f ".env" ]; then
    [ -f ".env.example" ] && cp .env.example .env && echo -e "${YELLOW}Da tao .env tu .env.example.${NC}" \
    || echo -e "${RED}Warning: Khong tim thay .env!${NC}"
else
    echo -e "${GREEN}.env da ton tai.${NC}"
fi
mkdir -p backend/data backend/tmp_uploads

# ==============================================================================
if [ "$MODE" = "test-ingest" ]; then
# ==============================================================================
    echo -e "\n${GREEN}========== CHE DO TEST INGESTION (500 VB Y TE) ==========${NC}"
    echo -e "${YELLOW}Config tu .env:${NC}"
    echo -e "  - Qdrant collection : $(grep TEST_QDRANT_COLLECTION .env | cut -d= -f2)"
    echo -e "  - Neo4j label prefix: $(grep TEST_NEO4J_LABEL_PREFIX .env | cut -d= -f2)"
    echo -e "  - Sample limit      : $(grep TEST_SAMPLE_LIMIT .env | cut -d= -f2)"
    echo -e "\n${CYAN}Bat dau pipeline... (log: .debug/ingestion_log_*.txt)${NC}"

    # Check Neo4j + Qdrant san sang truoc khi chay
    echo -e "${YELLOW}Cho 5s de DB containers khoi dong...${NC}"; sleep 5

    python -m backend.ingestion.chunking_embedding 2>&1 | tee /dev/null
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "\n${GREEN}Test ingestion HOAN TAT! Kiem tra ket qua:${NC}"
        echo -e "  - Qdrant: http://localhost:6337/dashboard"
        echo -e "  - Neo4j : http://localhost:7475"
        echo -e "  - Logs  : ${YELLOW}ls .debug/ingestion_log_*.txt | tail -1${NC}"
    else
        echo -e "\n${RED}Pipeline gap loi! Xem log de debug:${NC}"
        echo -e "  ${YELLOW}tail -100 \$(ls .debug/ingestion_log_*.txt | tail -1)${NC}"
    fi

# ==============================================================================
else
# ==============================================================================
    # --- 5. Kill old processes ---
    echo -e "\n${CYAN}[5] Don dep tien trinh cu (port 3005, 8005)...${NC}"
    for PORT in 3005 3006 8005 8006; do
        PID=$(lsof -t -i:$PORT 2>/dev/null)
        [ -n "$PID" ] && echo -e "${YELLOW}  Kill PID $PID (port $PORT)${NC}" && kill -9 $PID 2>/dev/null
    done
    sleep 2

    # --- 6. Start services ---
    echo -e "\n${CYAN}[6] Khoi dong Backend & Frontend...${NC}"
    nohup uvicorn backend.api.main:app --host 0.0.0.0 --port 8005 --reload --reload-dir=backend > backend_log.txt 2>&1 &
    echo -e "${YELLOW}Cho 20s de Backend warmup...${NC}"; sleep 20

    cd frontend || exit
    export PORT=3005
    nohup npm run dev -- --port 3005 > frontend_log.txt 2>&1 &
    cd ..

    echo -e "\n${GREEN}============================================================${NC}"
    echo -e "${GREEN} LEGAL-RAG DA SAN SANG!${NC}"
    echo -e "${GREEN}============================================================${NC}"
    echo -e "${CYAN}  - Frontend  : http://IP_MAY_CHU:3005${NC}"
    echo -e "${CYAN}  - Backend   : http://IP_MAY_CHU:8005${NC}"
    echo -e "${CYAN}  - Neo4j     : http://IP_MAY_CHU:7475${NC}"
    echo -e "${CYAN}  - Qdrant    : http://IP_MAY_CHU:6337/dashboard${NC}"
    echo -e "${YELLOW}Logs: tail -f backend_log.txt | tail -f frontend/frontend_log.txt${NC}"
fi

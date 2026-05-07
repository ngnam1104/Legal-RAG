#!/bin/bash

# ==============================================================================
# Legal-RAG: Script xem log dong thoi cho Backend & Frontend
# Su dung: ./tail_logs.sh
# ==============================================================================

LOG_DIR="logs"
LOG_BACKEND="$LOG_DIR/backend.log"
LOG_FRONTEND="$LOG_DIR/frontend.log"

if [ ! -d "$LOG_DIR" ]; then
    echo "LOI: Thu muc logs/ khong ton tai. Hay chay ./quick_start.sh truoc."
    exit 1
fi

echo "=============================================================================="
echo "DANG THEO DOI LOGS (Ctrl+C de thoat)"
echo "  - Backend : $LOG_BACKEND"
echo "  - Frontend: $LOG_FRONTEND"
echo "=============================================================================="

tail -f "$LOG_BACKEND" "$LOG_FRONTEND"

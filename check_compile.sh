#!/bin/bash
# check_compile.sh — Chạy sau khi git pull để kiểm tra syntax toàn bộ các file đã refactor
set -e

echo "=== [1/2] Git Pull ==="
git pull

echo ""
echo "=== [2/2] Syntax Check ==="
FILES=(
  "backend/config.py"
  "backend/ingestion/extractor/entities.py"
  "backend/agent/utils_legal.py"
  "backend/models/llm_client.py"
  "backend/database/neo4j_client.py"
  "utils/migrate_neo4j_relations.py"
  "utils/debug_neo4j_relations.py"
)

ALL_OK=true
for f in "${FILES[@]}"; do
  if python -m py_compile "$f" 2>&1; then
    echo "  OK   $f"
  else
    echo "  ERR  $f"
    ALL_OK=false
  fi
done

echo ""
if [ "$ALL_OK" = true ]; then
  echo "=== ALL FILES PASSED ==="
else
  echo "=== SOME FILES FAILED — check errors above ==="
  exit 1
fi

#!/usr/bin/env bash
# POST /v1/rag/query (FastMCP HTTP transport, default port 8000)
set -euo pipefail
BASE="${BASE:-http://localhost:8000}"

curl -sS -X POST "${BASE}/v1/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "what is taixing visa?",
    "collection_base": "taixing_knowledge",
    "request_id": "request_id_1",
    "session_id": "session_id_1",
    "k": 5,
    "k_max": 40
  }'
echo

curl -sS -X POST "${BASE}/v1/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "does taixing have paper A Novel Approach to Impedance-based Fault Location for High Voltage Cables?",
    "collection_base": "taixing_knowledge",
    "request_id": "request_id_1",
    "session_id": "session_id_1",
    "k": 5,
    "k_max": 40
  }'
echo

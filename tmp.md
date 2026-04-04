curl -X POST http://localhost:8000/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "what is taixing visa status",
    "collection_base": "taixing_knowledge",
    "request_id": "request_id_1",
    "session_id": "session_id_1",
    "k": 5,
    "k_max": 40
  }'

curl -X POST http://localhost:8000/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "does taixing have any publication?",
    "collection_base": "taixing_knowledge",
    "request_id": "request_id_1",
    "session_id": "session_id_1",
    "k": 5,
    "k_max": 40
  }'
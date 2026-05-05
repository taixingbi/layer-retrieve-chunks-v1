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


  python eva.py --base-url http://localhost:8000 -i eva/dataset/dataset-gold-test-1.0.0.json -o eva/result/dataset-gold-test-1.0.0.filled.json



curl -X POST http://localhost:8000/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "what is taixing linkedin",
    "collection_base": "taixing_knowledge",
    "request_id": "request_id_1",
    "session_id": "session_id_1",
    "k": 5,
    "k_max": 40
  }'


   {
    "input": "What is the link to Taixing's publication on impedance-based fault location?",
    "inference-output": "NOT_FOUND",
    "output": "https://www.researchgate.net/publication/261239443_A_novel_approach_to_impedance-based_fault_location_for_high_voltage_cables",

  }


  python eva/metric.py -i eva/result/dataset-gold-test-1.0.0.json -o eva/result/dataset-gold-test-eva-1.0.0.json





{
  "question": "asdasdadssdsdsd",
  "answer": "Taixing Bi's visa status is H4 EAD.",
  "citations": [
    {
      "id": "c1",
      "rank": 1,
      "chunk_id": "16310064034107487855",
      "source": "profile.json",
      "rrf_score": 0.91,
      "excerpt": "question: What is the candidate's visa status? answer: H4 EAD category: qa"
    }
  ],
  "meta": {
    "session_id": "asdasdasdasdasd",
    "request_id": "a3f7c2d1-8e4b-4f1a-b9c0-123456789abc",
    "latency_ms": 87,
    "model": "mistral-7b-instruct"
  }
}



curl -X POST http://localhost:8000/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What does H4 EAD permit Taixing to do in the US?",
    "collection_base": "taixing_knowledge",
    "request_id": "request_id_1",
    "session_id": "session_id_1",
    "k": 5
  }'..




  curl -sS -X POST http://192.168.86.179:30183/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is taixing visa",
    "collection_base": "taixing_knowledge",
    "request_id": "req-abc123",
    "session_id": "ses-xyz789",
    "k": 5,
    "k_max": 40
  }' | jq .
echo

  curl -sS -X POST http://192.168.86.179:30183/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What does H4 EAD permit Taixing?",
    "collection_base": "taixing_knowledge",
    "request_id": "req-abc123",
    "session_id": "ses-xyz789",
    "k": 5,
    "k_max": 40
  }' | jq .
echo


curl -sS -X POST http://192.168.86.179:30183/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Can Taixing work for any US company with an H4 EAD, or are there restrictions?",
    "collection_base": "taixing_knowledge",
    "request_id": "req-abc123",
    "session_id": "ses-xyz789",
    "k": 5,
    "k_max": 40
  }' | jq .
echo


curl -sS -X POST http://192.168.86.179:30183/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "what is taixing visa",
    "collection_base": "taixing_knowledge",
    "request_id": "req-abc123",
    "session_id": "ses-xyz789",
    "k": 5,
    "k_max": 40
  }'

curl -sS -X POST http://192.168.86.179:30183/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Can Taixing work for any US company with an H4 EAD, or are there restrictions?",
    "collection_base": "taixing_knowledge",
    "request_id": "req-abc123",
    "session_id": "ses-xyz789",
    "k": 5,
    "k_max": 40,
    "include_retrieval_hits": true
  }' | jq .
echo
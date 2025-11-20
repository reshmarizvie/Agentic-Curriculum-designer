#!/usr/bin/env bash
set -euo pipefail
export $(grep -v '^#' .env | xargs -d '\n' -I{} echo {} )

# Start Curriculum Agent (Gemma 2B)
uvicorn apps.curriculum_agent.main:app --host 0.0.0.0 --port 8001 &
PID1=$!

# Start IR Agent
uvicorn apps.ir_agent.main:app --host 0.0.0.0 --port 8002 &
PID2=$!

# Start NLP Agent
uvicorn apps.nlp_agent.main:app --host 0.0.0.0 --port 8003 &
PID3=$!

# Start Orchestrator
uvicorn apps.orchestrator.main:app --host 0.0.0.0 --port 8000 &
PID4=$!

echo "Agents running: $PID1 $PID2 $PID3; Orchestrator: $PID4"
wait

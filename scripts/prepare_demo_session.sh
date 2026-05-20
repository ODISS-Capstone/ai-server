#!/usr/bin/env bash
# ODISS 데모 세션 사전 점검 (GPU 서버에서 실행)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

GPU_URL="${ODISS_CLOUD_URL:-http://127.0.0.1:8000}"
JETSON_IP="${LOCAL_Agent_IP:-192.168.0.73}"
DEMO_SPEAKER="${ODISS_DEMO_SPEAKER_ID:-demo_kimyoungsu}"

echo "=== ODISS Demo preflight ==="
echo "GPU backend: $GPU_URL"
echo "Jetson IP:   $JETSON_IP"
echo "Demo speaker: $DEMO_SPEAKER"
echo

check_http() {
  local url="$1"
  local label="$2"
  if code=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 3 "$url"); then
  if [[ "$code" == "200" ]]; then
    echo "[OK] $label ($url)"
  else
    echo "[WARN] $label HTTP $code ($url)"
  fi
  else
    echo "[FAIL] $label unreachable ($url)"
  fi
}

check_http "$GPU_URL/health" "ai-server health"
check_http "$GPU_URL/health/llm" "LLM health"

if command -v ping >/dev/null 2>&1; then
  if ping -c1 -W2 "$JETSON_IP" >/dev/null 2>&1; then
    echo "[OK] Jetson ping $JETSON_IP"
  else
    echo "[WARN] Jetson ping failed $JETSON_IP"
  fi
fi

echo
echo "--- Reset demo speaker (optional) ---"
echo "  python3 scripts/reset_demo_speaker.py --speaker-id $DEMO_SPEAKER"
echo
echo "--- GPU: start ai-server + vLLM if needed ---"
echo "  cd $ROOT && source .venv/bin/activate  # if used"
echo
echo "--- Jetson: local_agent ---"
echo "  cd ~/local_agent && source .env"
echo "  export ODISS_SPEAKER_ID=$DEMO_SPEAKER"
echo "  PYTHONPATH=src python3 -m src.main"
echo
echo "--- Human runbook ---"
echo "  scripts/odiss_demo_story_human_runbook.md"
echo

#!/usr/bin/env bash
# GPU 서버(192.168.0.12)의 ai-server :8000 을 Jetson localhost:18000 으로 노출.
# Jetson 방화벽/라우팅으로 8000 직접 접속이 안 될 때 사용.
# 사용: ./scripts/start_jetson_cloud_tunnel.sh
# Jetson .env: ODISS_CLOUD_URL=http://127.0.0.1:18000

set -euo pipefail
JETSON_HOST="${JETSON_HOST:-jepetoleee@192.168.0.73}"
REMOTE_PORT="${REMOTE_PORT:-18000}"
LOCAL_PORT="${LOCAL_PORT:-8000}"

if pgrep -f "ssh -N.*${REMOTE_PORT}:127.0.0.1:${LOCAL_PORT}" >/dev/null; then
  echo "Tunnel already running."
  exit 0
fi

exec ssh -N \
  -o StrictHostKeyChecking=no \
  -o ServerAliveInterval=30 \
  -R "${REMOTE_PORT}:127.0.0.1:${LOCAL_PORT}" \
  "${JETSON_HOST}"

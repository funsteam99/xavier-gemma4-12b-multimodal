#!/usr/bin/env bash
set -euo pipefail

# Disable CUDA VMM memory pool to prevent out of memory during multimodal audio/image encoding on Tegra
export GGML_CUDA_NO_VMM=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-/home/nvidia/src/llama.cpp}"
LLAMA_SERVER="${LLAMA_SERVER:-$LLAMA_CPP_DIR/build/bin/llama-server}"
MODEL_DIR="${MODEL_DIR:-/media/nvidia/sd/models/gemma-4-12b-qat-mtp}"
MODEL="${MODEL:-$MODEL_DIR/gemma-4-12B-it-QAT-Q4_0.gguf}"
DRAFT_MODEL="${DRAFT_MODEL:-$MODEL_DIR/gemma-4-12B-it-qat-assistant-MTP-Q8_0.gguf}"
MMPROJ="${MMPROJ:-$MODEL_DIR/mmproj-gemma-4-12B-it-QAT-BF16.gguf}"
CONSOLE_TARGET="${CONSOLE_TARGET:-/media/nvidia/sd/gemma4-12b-qat-mtp-console}"
BACKEND_PORT="${BACKEND_PORT:-18085}"
CONSOLE_PORT="${CONSOLE_PORT:-18091}"
BACKEND_LOG="${BACKEND_LOG:-/tmp/llama_gemma4_12b_qat_mtp.log}"
CONSOLE_LOG="${CONSOLE_LOG:-/tmp/gemma4_12b_qat_mtp_console.log}"
APP_TITLE="${APP_TITLE:-Gemma 4 12B QAT + MTP Console}"
MODEL_HINT="${MODEL_HINT:-gemma-4-12B-it-QAT-Q4_0.gguf + MTP Q8_0}"
CTX_SIZE="${CTX_SIZE:-4096}"
BATCH_SIZE="${BATCH_SIZE:-512}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
DRAFT_GPU_LAYERS="${DRAFT_GPU_LAYERS:-99}"
SPEC_DRAFT_N_MAX="${SPEC_DRAFT_N_MAX:-4}"
IMAGE_MAX_TOKENS="${IMAGE_MAX_TOKENS:-768}"
MIN_AVAILABLE_MB="${MIN_AVAILABLE_MB:-10240}"
CACHE_TYPE_K="${CACHE_TYPE_K:-q4_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-q4_0}"
CACHE_TYPE_K_DRAFT="${CACHE_TYPE_K_DRAFT:-q4_0}"
CACHE_TYPE_V_DRAFT="${CACHE_TYPE_V_DRAFT:-q4_0}"

BACKEND_ARGS=(
  -m "$MODEL"
  --model-draft "$DRAFT_MODEL"
  --spec-type draft-mtp
  --spec-draft-n-max "$SPEC_DRAFT_N_MAX"
  --mmproj "$MMPROJ"
  --media-path /tmp
  --host 0.0.0.0
  --port "$BACKEND_PORT"
  -c "$CTX_SIZE"
  -b "$BATCH_SIZE"
  -ub "$UBATCH_SIZE"
  -ngl "$N_GPU_LAYERS"
  -ngld "$DRAFT_GPU_LAYERS"
  --threads 3
  --parallel 1
  --image-max-tokens "$IMAGE_MAX_TOKENS"
  --cache-type-k "$CACHE_TYPE_K"
  --cache-type-v "$CACHE_TYPE_V"
  --cache-type-k-draft "$CACHE_TYPE_K_DRAFT"
  --cache-type-v-draft "$CACHE_TYPE_V_DRAFT"
  --cache-ram 0
  --reasoning off
  --no-warmup
)

die() {
  echo "error: $*" >&2
  exit 1
}

need_file() {
  [ -f "$1" ] || die "missing file: $1"
}

memory_check() {
  echo "memory:"
  free -h
  echo
  timeout 2 tegrastats --interval 1000 2>/dev/null | head -1 || true

  local available_mb
  available_mb="$(awk '/MemAvailable:/ { print int($2 / 1024) }' /proc/meminfo)"
  [ "$available_mb" -ge "$MIN_AVAILABLE_MB" ] \
    || die "only ${available_mb} MiB available; require at least ${MIN_AVAILABLE_MB} MiB"
}

port_pid() {
  local port="$1"
  ss -ltnp 2>/dev/null \
    | awk -v port=":$port " '$0 ~ port { print $NF }' \
    | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' \
    | head -1
}

stop_if_matching() {
  local port="$1"
  local pattern="$2"
  local pid
  pid="$(port_pid "$port" || true)"
  [ -n "$pid" ] || return 0

  local args
  args="$(ps -p "$pid" -o args= || true)"
  case "$args" in
    *"$pattern"*)
      echo "stopping pid $pid on port $port"
      kill "$pid"
      sleep 2
      ;;
    *)
      die "port $port is occupied by unrelated process: pid=$pid args=$args"
      ;;
  esac
}

check() {
  echo "checking Xavier Gemma 4 12B QAT + MTP prerequisites"
  memory_check
  need_file "$LLAMA_SERVER"
  need_file "$MODEL"
  need_file "$DRAFT_MODEL"
  need_file "$MMPROJ"
  need_file "$ROOT_DIR/console/server.py"
  need_file "$ROOT_DIR/console/public/index.html"
  command -v python3 >/dev/null || die "python3 not found"
  command -v curl >/dev/null || die "curl not found"
  command -v ss >/dev/null || die "ss not found"

  echo
  "$LLAMA_SERVER" --version 2>&1 || true
  local server_help
  server_help="$("$LLAMA_SERVER" --help 2>&1 || true)"
  grep -q -- "--spec-type" <<<"$server_help" \
    || die "llama-server does not expose speculative decoding options"

  echo
  echo "model:   $MODEL"
  echo "draft:   $DRAFT_MODEL"
  echo "mmproj:  $MMPROJ"
  echo "console: $CONSOLE_TARGET"
  echo "ports:   backend=$BACKEND_PORT console=$CONSOLE_PORT"
  echo "limits:  ctx=$CTX_SIZE batch=$BATCH_SIZE ubatch=$UBATCH_SIZE draft_n=$SPEC_DRAFT_N_MAX image_max_tokens=$IMAGE_MAX_TOKENS cache_k=$CACHE_TYPE_K cache_v=$CACHE_TYPE_V cache_kd=$CACHE_TYPE_K_DRAFT cache_vd=$CACHE_TYPE_V_DRAFT"
}

install_console() {
  echo "installing console to $CONSOLE_TARGET"
  mkdir -p "$CONSOLE_TARGET/public"
  cp "$ROOT_DIR/console/server.py" "$CONSOLE_TARGET/server.py"
  cp "$ROOT_DIR/console/public/index.html" "$CONSOLE_TARGET/public/index.html"
  cp "$ROOT_DIR/console/public/styles.css" "$CONSOLE_TARGET/public/styles.css"
  cp "$ROOT_DIR/console/public/app.js" "$CONSOLE_TARGET/public/app.js"
}

start_backend() {
  check
  stop_if_matching "$BACKEND_PORT" "llama-server"

  echo "starting QAT + MTP llama-server on port $BACKEND_PORT"
  setsid "$LLAMA_SERVER" "${BACKEND_ARGS[@]}" \
    </dev/null >"$BACKEND_LOG" 2>&1 &

  for _ in $(seq 1 60); do
    if curl -fsS "http://127.0.0.1:$BACKEND_PORT/health" >/tmp/gemma4_12b_health.json 2>/dev/null; then
      echo "backend healthy:"
      cat /tmp/gemma4_12b_health.json
      echo
      return 0
    fi
    sleep 2
  done

  echo "backend did not become healthy; recent log:"
  tail -120 "$BACKEND_LOG" || true
  exit 1
}

start_console() {
  install_console
  stop_if_matching "$CONSOLE_PORT" "$CONSOLE_TARGET/server.py"

  echo "starting console on port $CONSOLE_PORT"
  PORT="$CONSOLE_PORT" \
    GEMMA_BACKEND="http://127.0.0.1:$BACKEND_PORT" \
    APP_TITLE="$APP_TITLE" \
    MODEL_HINT="$MODEL_HINT" \
    setsid python3 "$CONSOLE_TARGET/server.py" \
    </dev/null >"$CONSOLE_LOG" 2>&1 &

  sleep 1
  curl -fsS -X POST -H "Content-Type: application/json" -d "{}" \
    "http://127.0.0.1:$CONSOLE_PORT/api/health"
  echo
}

status() {
  echo "ports:"
  ss -ltnp | grep -E ":($BACKEND_PORT|$CONSOLE_PORT) " || true

  echo
  echo "backend health:"
  curl -fsS "http://127.0.0.1:$BACKEND_PORT/health" || true
  echo

  echo
  echo "console health:"
  curl -fsS -X POST -H "Content-Type: application/json" -d "{}" \
    "http://127.0.0.1:$CONSOLE_PORT/api/health" || true
  echo

  echo
  echo "logs:"
  echo "$BACKEND_LOG"
  echo "$CONSOLE_LOG"
}

stop_all() {
  stop_if_matching "$CONSOLE_PORT" "$CONSOLE_TARGET/server.py"
  stop_if_matching "$BACKEND_PORT" "llama-server"
}

usage() {
  cat <<EOF
Usage: $0 <command>

Commands:
  check             Validate paths, model files, and llama.cpp CUDA build info
  install-console   Copy console files into \$CONSOLE_TARGET
  start-backend     Start/restart 12B QAT + MTP llama-server
  start-console     Install and start/restart the QAT + MTP browser console
  start             Start/restart backend and console
  status            Show ports, health checks, and logs
  stop              Stop backend and console if they match this setup

Environment overrides:
  LLAMA_CPP_DIR=$LLAMA_CPP_DIR
  MODEL=$MODEL
  DRAFT_MODEL=$DRAFT_MODEL
  MMPROJ=$MMPROJ
  CONSOLE_TARGET=$CONSOLE_TARGET
  BACKEND_PORT=$BACKEND_PORT
  CONSOLE_PORT=$CONSOLE_PORT
  CTX_SIZE=$CTX_SIZE
  BATCH_SIZE=$BATCH_SIZE
  UBATCH_SIZE=$UBATCH_SIZE
  SPEC_DRAFT_N_MAX=$SPEC_DRAFT_N_MAX
  MIN_AVAILABLE_MB=$MIN_AVAILABLE_MB
  IMAGE_MAX_TOKENS=$IMAGE_MAX_TOKENS
  CACHE_TYPE_K=$CACHE_TYPE_K
  CACHE_TYPE_V=$CACHE_TYPE_V
  CACHE_TYPE_K_DRAFT=$CACHE_TYPE_K_DRAFT
  CACHE_TYPE_V_DRAFT=$CACHE_TYPE_V_DRAFT
EOF
}

cmd="${1:-start}"
case "$cmd" in
  check) check ;;
  install-console) install_console ;;
  start-backend) start_backend ;;
  start-console) start_console ;;
  start) start_backend; start_console; status ;;
  status) status ;;
  stop) stop_all ;;
  -h|--help|help) usage ;;
  *) usage; die "unknown command: $cmd" ;;
esac

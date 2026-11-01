#!/bin/bash
# AirMind: one command to start everything.
#   ./start.sh            -> Arabic
#   ./start.sh --lang en  -> English (any airmind.py option works here)

MODEL_DIR="$HOME/.lmstudio/models/lmstudio-community/gemma-4-26B-A4B-it-GGUF"
MODEL="$MODEL_DIR/gemma-4-26B-A4B-it-Q4_K_M.gguf"
MMPROJ="$MODEL_DIR/mmproj-gemma-4-26B-A4B-it-BF16.gguf"
# the project's own Python environment (made by ./setup.sh); the old haunted-cam one still works as a fallback
VENV="$(cd "$(dirname "$0")" && pwd)/.venv"
[ -d "$VENV" ] || VENV="$HOME/Documents/AI/haunted-cam/.venv"
# How many of the model's layers keep their "experts" on the CPU. Lower = faster answers but more
# GPU memory; too low and macOS audio can stop. 12 is safe, 8 tested OK; try 6 or 4:  CPU_MOE=4 ./start.sh
CPU_MOE="${CPU_MOE:-8}"
# Optional: fewer image tokens = faster reading of the picture (try IMG_TOKENS=256)
EXTRA=""
[ -n "$IMG_TOKENS" ] && EXTRA="--image-max-tokens $IMG_TOKENS"
cd "$(dirname "$0")"

# 1) keep the Mac's default GPU memory limit. Raising it (iogpu.wired_limit_mb) let the model take
#    almost all the RAM and macOS audio stopped working ("AudioQueueStart failed"). Instead part of
#    the model runs on the CPU (--n-cpu-moe), which fits in the default limit.
if [ "$(sysctl -n iogpu.wired_limit_mb)" != "0" ]; then
  echo "Putting the GPU memory limit back to the Mac default (needs your password):"
  sudo sysctl iogpu.wired_limit_mb=0
fi

# 2) start the model. Two choices:
#    - default: Gemma 4 26B through llama-server (big, smart, slower on 24 GB)
#    - FAST=1:  a small vision model through Ollama (much faster; download once with
#               `ollama pull qwen3-vl:4b` or `ollama pull qwen3-vl:2b`). Pick one with OLLAMA_MODEL=name
SERVER_PID=""
APP_ARGS=()
if [ -n "$FAST" ] || [ -n "$OLLAMA_MODEL" ]; then
  # pick with OLLAMA_MODEL=name; otherwise the best one you have (run `python bench.py` to compare them)
  if ! curl -s http://127.0.0.1:11434/api/tags >/dev/null; then
    echo "Starting Ollama..."
    ollama serve > /tmp/ollama.log 2>&1 &
    SERVER_PID=$!
    until curl -s http://127.0.0.1:11434/api/tags >/dev/null; do sleep 1; done
  fi
  if [ -z "$OLLAMA_MODEL" ]; then            # the best small vision model you have downloaded
    for m in qwen3-vl:4b qwen3-vl:2b qwen2.5vl:3b; do
      if ollama list | grep -q "^$m"; then OLLAMA_MODEL=$m; break; fi
    done
    OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5vl:3b}"
  fi
  if ! ollama list | grep -q "^${OLLAMA_MODEL%%:*}"; then
    echo "Model $OLLAMA_MODEL isn't downloaded yet. Run:  ollama pull $OLLAMA_MODEL"; exit 1
  fi
  curl -s http://127.0.0.1:8080/health >/dev/null && echo "(note: llama-server is also running; using $OLLAMA_MODEL)"
  APP_ARGS=(--model "$OLLAMA_MODEL")
  echo "Using $OLLAMA_MODEL via Ollama."
elif ! curl -s http://127.0.0.1:8080/health >/dev/null; then
  echo "Starting Gemma 4 with CPU_MOE=$CPU_MOE (takes about 30 seconds)..."
  llama-server -m "$MODEL" --mmproj "$MMPROJ" -c 4096 -np 1 --reasoning-budget 0 --n-cpu-moe "$CPU_MOE" --load-mode none $EXTRA --port 8080 \
    > /tmp/llama-server.log 2>&1 &
  SERVER_PID=$!
  trap '[ -n "$SERVER_PID" ] && kill $SERVER_PID 2>/dev/null' EXIT
  until curl -s http://127.0.0.1:8080/health | grep -q ok; do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
      echo "The model server stopped. Last lines of /tmp/llama-server.log:"; tail -15 /tmp/llama-server.log; exit 1
    fi
    sleep 1
  done
  echo "Model ready."
fi

# 3) speech helper: `say` must be started from the shell, not from inside the camera app
SPEAKER_PID=""
if ! curl -s http://127.0.0.1:8765/ >/dev/null; then
  python3 speaker.py > /tmp/airmind-speaker.log 2>&1 &
  SPEAKER_PID=$!
fi
trap '[ -n "$SERVER_PID" ] && kill $SERVER_PID 2>/dev/null; [ -n "$SPEAKER_PID" ] && kill $SPEAKER_PID 2>/dev/null' EXIT

# 4) run the app
source "$VENV/bin/activate"
python airmind.py "${APP_ARGS[@]}" "$@"

#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
for gpu in $(seq 0 7); do
  session="zofo_rebuttal_gpu${gpu}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "[skip] tmux session already exists: $session"
  else
    tmux new-session -d -s "$session" "cd '$ROOT' && exec python rebuttal/worker.py --gpu $gpu --queue rebuttal/queues/gpu_${gpu}.txt"
    echo "[started] $session"
  fi
done
if ! tmux has-session -t zofo_rebuttal_report 2>/dev/null; then
  tmux new-session -d -s zofo_rebuttal_report "cd '$ROOT' && exec python rebuttal/aggregate_results.py --watch --interval 600"
  echo "[started] zofo_rebuttal_report"
fi

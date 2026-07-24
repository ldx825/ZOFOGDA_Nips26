#!/usr/bin/env bash
set -euo pipefail
for gpu in $(seq 0 7); do tmux kill-session -t "zofo_rebuttal_gpu${gpu}" 2>/dev/null || true; done
tmux kill-session -t zofo_rebuttal_report 2>/dev/null || true

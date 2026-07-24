#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
echo "=== tmux ==="
tmux list-sessions 2>/dev/null | grep 'zofo_rebuttal' || true
echo "=== workers ==="
for f in "$ROOT"/rebuttal/logs/worker_gpu_*_state.json; do
  [ -e "$f" ] || continue
  python - "$f" <<'PY2'
import json,sys
p=sys.argv[1]; d=json.load(open(p)); print(f"GPU {d['gpu']}: {d['status']} run={d.get('run_id','-')} {d.get('index','-')}/{d.get('total','-')} ok={d.get('completed_this_worker',0)} fail={d.get('failed',0)}")
PY2
done
echo "=== result counts ==="
python - <<'PY2'
import json
from pathlib import Path
c={}
for p in Path('rebuttal/results').glob('*/summary.json'):
 try:s=json.load(open(p)).get('status','unknown')
 except:s='broken'
 c[s]=c.get(s,0)+1
print(c)
PY2

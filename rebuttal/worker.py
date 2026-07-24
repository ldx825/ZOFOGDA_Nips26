#!/usr/bin/env python3
"""Sequential GPU worker with resume/skip support."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_summary(config):
    out = ROOT / config["output_dir"]
    path = out / "summary.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--queue", required=True)
    args = ap.parse_args()
    queue_path = ROOT / args.queue
    configs = [line.strip() for line in queue_path.read_text().splitlines() if line.strip()]
    state_path = ROOT / "rebuttal" / "logs" / f"worker_gpu_{args.gpu}_state.json"
    worker_log = ROOT / "rebuttal" / "logs" / f"worker_gpu_{args.gpu}.log"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    completed = skipped = failed = 0
    with worker_log.open("a", encoding="utf-8") as wlog:
        for index, config_rel in enumerate(configs, 1):
            config_path = ROOT / config_rel
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            summary = load_summary(cfg)
            if summary and summary.get("status") == "completed":
                skipped += 1
                continue
            state = {
                "gpu": args.gpu, "status": "running", "queue": args.queue,
                "index": index, "total": len(configs), "run_id": cfg["run_id"],
                "phase": cfg["phase"], "completed_this_worker": completed,
                "skipped": skipped, "failed": failed,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            run_log = ROOT / "rebuttal" / "logs" / f"{cfg['run_id']}.log"
            msg = f"[{time.strftime('%F %T')}] GPU {args.gpu} START {index}/{len(configs)} {cfg['run_id']}\n"
            wlog.write(msg); wlog.flush(); print(msg, end="", flush=True)
            with run_log.open("a", encoding="utf-8") as out:
                proc = subprocess.run(
                    [sys.executable, str(ROOT / "rebuttal" / "run_experiment.py"),
                     "--config", str(config_path), "--device", f"cuda:{args.gpu}"],
                    cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT,
                )
            if proc.returncode == 0:
                completed += 1
                result = "OK"
            else:
                failed += 1
                result = f"FAILED({proc.returncode})"
            msg = f"[{time.strftime('%F %T')}] GPU {args.gpu} {result} {cfg['run_id']}\n"
            wlog.write(msg); wlog.flush(); print(msg, end="", flush=True)
    final = {
        "gpu": args.gpu, "status": "finished", "queue": args.queue,
        "total": len(configs), "completed_this_worker": completed,
        "skipped": skipped, "failed": failed,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    state_path.write_text(json.dumps(final, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

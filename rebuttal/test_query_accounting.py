#!/usr/bin/env python3
"""Fast regression tests for the rebuttal runner's query accounting."""
from __future__ import annotations
import json
import tempfile
from dataclasses import replace
from pathlib import Path

from run_experiment import Config, Trainer


def run_case(base: Config, name: str, expected_step_cost: int, steps: int = 2):
    cfg = replace(
        base,
        run_id=f"query_test_{name}",
        max_train_forward_queries=expected_step_cost * steps,
        eval_every_forward_queries=expected_step_cost,
    )
    with tempfile.TemporaryDirectory(prefix="zofo_query_test_") as tmp:
        trainer = Trainer(cfg)
        assert trainer.expected_step_cost() == expected_step_cost
        summary = trainer.run(Path(tmp) / name)
    assert summary["global_steps"] == steps, (name, summary)
    assert summary["cum_train_forward_queries"] == expected_step_cost * steps, (name, summary)
    return summary


def main():
    auc = Config(
        task="auc", dataset="svmguide1", device="cpu", hidden=8, batch_size=32,
        standardize=False, eval_train_batches=1, seed=7, data_seed=0,
    )
    cases = []
    for m in (1, 5, 15):
        cases.append(run_case(replace(auc, M=m, estimator="two_sided"), f"two_sided_M{m}", 2*m+1))
    cases.append(run_case(replace(auc, M=5, estimator="one_sided"), "one_sided_M5", 7))
    cases.append(run_case(replace(auc, estimator="direct"), "zofo_direct", 1))
    cases.append(run_case(replace(auc, estimator="two_sided", x_momentum=False, y_momentum=False), "no_momentum", 11))
    robust = replace(auc, task="robust", dataset="australian", method="ttgda", estimator="direct",
                     gamma=1.0, lam=1.0, theta1=1e-3)
    cases.append(run_case(robust, "robust_fo", 1))
    print(json.dumps({
        "status": "passed",
        "cases": len(cases),
        "accounting": {
            "two_sided": "2*M+1",
            "one_sided": "M+2",
            "direct_or_FO": "1",
        },
    }, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""One time-vs-AUC figure per dataset for six rebuttal conditions."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

import manage
import plot_results


SPECS = [
    ("TTGDA-original", "TTGDA", "baseline", "paper", "#2ca02c", "-"),
    ("TTSGDA-original", "TTSGDA", "baseline", "paper", "#2345ff", "-"),
    ("ZOFOGDA-original", "ZOFOSGDA", "baseline", "paper", "#ff2d2d", "-"),
    ("ZOFOGDA-no-x", "ZOFOSGDA", "momentum", "no_x", "#9467bd", "--"),
    ("ZOFOGDA-no-y", "ZOFOSGDA", "momentum", "no_y", "#8c564b", "-."),
    ("ZOFOGDA-no-both", "ZOFOSGDA", "momentum", "no_both", "#17becf", ":"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default=",".join(manage.DATASETS))
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "plots" / "time_accuracy_six"),
    )
    args = parser.parse_args()

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lookup = plot_results.manifest_lookup()

    for dataset in datasets:
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        for label, method, experiment, value, color, linestyle in SPECS:
            curves = []
            for seed in manage.SEEDS:
                run = plot_results.resolve_key(
                    lookup, dataset, method, experiment, value, seed
                )
                curve = plot_results.run_curve(run, "time")
                if curve is not None:
                    curves.append(curve)
            aggregate = plot_results.aggregate(curves, "time")
            if aggregate is None:
                raise RuntimeError(
                    f"{dataset}/{label}: expected three complete time curves"
                )
            time_ms, mean_auc, std_auc = aggregate
            ax.plot(
                time_ms,
                mean_auc,
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=2.0,
            )
            ax.fill_between(
                time_ms,
                mean_auc - std_auc,
                mean_auc + std_auc,
                color=color,
                alpha=0.12,
                linewidth=0,
            )

        ax.set_title(dataset.capitalize(), fontsize=14)
        ax.set_xlabel("Time (ms)", fontsize=12)
        ax.set_ylabel("Test AUC (%)", fontsize=12)
        ax.grid(True, alpha=0.25, linewidth=0.7)
        ax.legend(fontsize=8.5, frameon=True, ncol=2)
        fig.tight_layout()
        stem = output_dir / f"{dataset}_time_accuracy_six"
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {stem.with_suffix('.pdf')} and {stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()

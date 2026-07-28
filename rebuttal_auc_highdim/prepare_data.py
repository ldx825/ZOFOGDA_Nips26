#!/usr/bin/env python3
"""Convert the official Arcene and Dexter train/validation files to LIBSVM."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "rebuttal_auc_highdim" / "source_data"
OUTPUT = ROOT / "26.1.29.ZOFOGDA_AUC" / "data_highdim"


def labels(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not values or set(values) != {"-1", "1"}:
        raise ValueError(f"Unexpected labels in {path}: {set(values)}")
    return values


def convert_arcene(data_path: Path, label_path: Path, output_path: Path) -> None:
    ys = labels(label_path)
    rows = data_path.read_text().splitlines()
    if len(rows) != len(ys):
        raise ValueError(f"{data_path}: {len(rows)} rows but {len(ys)} labels")
    with output_path.open("w") as out:
        for y, row in zip(ys, rows):
            values = row.split()
            if len(values) != 10_000:
                raise ValueError(f"{data_path}: expected 10000 features, got {len(values)}")
            fields = [f"{i}:{value}" for i, value in enumerate(values, 1) if value != "0"]
            out.write(y + " " + " ".join(fields) + "\n")


def convert_dexter(data_path: Path, label_path: Path, output_path: Path) -> None:
    ys = labels(label_path)
    rows = data_path.read_text().splitlines()
    if len(rows) != len(ys):
        raise ValueError(f"{data_path}: {len(rows)} rows but {len(ys)} labels")
    with output_path.open("w") as out:
        for y, row in zip(ys, rows):
            converted = []
            for field in row.split():
                index_text, value = field.split(":", 1)
                # The official Dexter sparse files use zero-based feature indices.
                index = int(index_text) + 1
                if not 1 <= index <= 20_000:
                    raise ValueError(f"{data_path}: invalid feature index {index_text}")
                converted.append(f"{index}:{value}")
            out.write(y + " " + " ".join(converted) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    arcene = SOURCE / "arcene" / "ARCENE"
    dexter = SOURCE / "dexter" / "DEXTER"
    jobs = [
        (convert_arcene, arcene / "arcene_train.data", arcene / "arcene_train.labels",
         args.output / "arcene"),
        (convert_arcene, arcene / "arcene_valid.data", SOURCE / "arcene" / "arcene_valid.labels",
         args.output / "arcene.t"),
        (convert_dexter, dexter / "dexter_train.data", dexter / "dexter_train.labels",
         args.output / "dexter"),
        (convert_dexter, dexter / "dexter_valid.data", SOURCE / "dexter" / "dexter_valid.labels",
         args.output / "dexter.t"),
    ]
    for converter, data_path, label_path, output_path in jobs:
        converter(data_path, label_path, output_path)
        print(f"{output_path}: sha256={sha256(output_path)}")


if __name__ == "__main__":
    main()

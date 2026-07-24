import os
import numpy as np
from sklearn.datasets import load_svmlight_file

def compute_imbalance_rate(path: str):
    """
    Compute imbalance statistics for a LIBSVM/SVMLight file.
    Returns:
        n: number of instances
        pos_ratio: proportion of +1 (or the larger label if not exactly +/-1)
        neg_ratio: proportion of -1 (or the other label)
        minority_ratio: min(pos_ratio, neg_ratio)
        majority_ratio: max(pos_ratio, neg_ratio)
        ratio_majority_to_minority: majority_ratio / minority_ratio
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    _, y = load_svmlight_file(path)
    y = np.asarray(y)

    # Count each label
    labels, counts = np.unique(y, return_counts=True)
    n = int(counts.sum())

    if len(labels) != 2:
        raise ValueError(f"Expected binary labels, got {len(labels)} labels: {labels.tolist()}")

    # Define two classes
    c0, c1 = counts[0], counts[1]
    r0, r1 = c0 / n, c1 / n

    minority_ratio = float(min(r0, r1))
    majority_ratio = float(max(r0, r1))
    ratio_majority_to_minority = float(majority_ratio / minority_ratio) if minority_ratio > 0 else np.inf

    return {
        "file": path,
        "n_instances": n,
        "labels": labels.tolist(),
        "counts": counts.tolist(),
        "ratios": [float(r0), float(r1)],
        "minority_ratio (Imbalance Rate)": minority_ratio,
        "majority_ratio": majority_ratio,
        "majority/minority": ratio_majority_to_minority
    }

def summarize(paths):
    for p in paths:
        stats = compute_imbalance_rate(p)
        print("=" * 70)
        print(f"File: {stats['file']}")
        print(f"Instances: {stats['n_instances']}")
        print(f"Labels: {stats['labels']}")
        print(f"Counts: {stats['counts']}")
        print(f"Class ratios: {stats['ratios'][0]:.6f}, {stats['ratios'][1]:.6f}")
        print(f"Minority ratio (Imbalance Rate): {stats['minority_ratio (Imbalance Rate)']:.6f}")
        print(f"Majority/minority: {stats['majority/minority']:.3f}")

if __name__ == "__main__":
    # 修改成你的真实路径
    train_path = "./data/svmguide1"
    test_path  = "./data/svmguide1.t"

    paths = [train_path]
    if os.path.exists(test_path):
        paths.append(test_path)

    summarize(paths)

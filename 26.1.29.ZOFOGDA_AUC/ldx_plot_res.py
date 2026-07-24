import os
import csv
import argparse
import matplotlib.pyplot as plt


def read_log(csv_path):
    """
    Reads a CSV log file with columns: epoch, train_loss, test_auc
    (Your training scripts log exactly these keys.)
    Also tolerates a possible config row like: ['#cfg', '{...}'] by skipping rows
    that don't contain required fields.
    """
    epochs, train_obj, test_auc = [], [], []

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Log not found: {csv_path}")

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip non-standard rows (e.g., cfg dump)
            if row is None:
                continue
            if "epoch" not in row or "train_loss" not in row or "test_auc" not in row:
                continue
            try:
                epochs.append(int(row["epoch"]))
                train_obj.append(float(row["train_loss"]))
                test_auc.append(float(row["test_auc"]))
            except Exception:
                # Skip malformed lines
                continue

    if len(epochs) == 0:
        raise ValueError(f"No valid rows parsed from: {csv_path}")

    return epochs, train_obj, test_auc


def resolve_log_path(log_dir: str, log_prefix: str, dataset: str) -> str:
    # matches your CLI convention: {log_prefix}_{dataset}_log.csv
    return os.path.join(log_dir, f"{log_prefix}_{dataset}_log.csv")


def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def main():
    parser = argparse.ArgumentParser("Plot comparison curves from TTGDA/TTSGDA/ZOFOGDA logs")
    parser.add_argument("--dataset", type=str, default="svmguide1",
                        help="Dataset name used in log file stem.")
    parser.add_argument("--log_dir", type=str, default="./result",
                        help="Directory containing *_log.csv files.")
    parser.add_argument("--img_dir", type=str, default="./img",
                        help="Directory to save images.")

    # prefixes should match what your training CLI used in --log_prefix
    parser.add_argument("--ttsgda_prefix", type=str, default="ttsgda_auc",
                        help="Log prefix for TTSGDA.")
    parser.add_argument("--ttgda_prefix", type=str, default="ttgda_auc",
                        help="Log prefix for TTGDA.")
    parser.add_argument("--zofogda_prefix", type=str, default="zofoGDA_auc",
                        help="Log prefix for ZOFOGDA.")

    parser.add_argument("--title_suffix", type=str, default="",
                        help="Optional suffix appended to plot titles.")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--auc_as_percent", action="store_true",
                        help="If set, plot AUC in percent (x100). Default: raw [0,1].")

    parser.add_argument("--skip_missing", action="store_true",
                        help="If set, skip missing logs instead of erroring.")
    args = parser.parse_args()

    ensure_dir(args.img_dir)

    paths = {
        "TTSGDA": resolve_log_path(args.log_dir, args.ttsgda_prefix, args.dataset),
        "ZOFOGDA": resolve_log_path(args.log_dir, args.zofogda_prefix, args.dataset),
        "TTGDA": resolve_log_path(args.log_dir, args.ttgda_prefix, args.dataset),
    }

    curves = {}
    for name, p in paths.items():
        if not os.path.exists(p):
            if args.skip_missing:
                print(f"[WARN] Missing log, skipping: {p}")
                continue
            raise FileNotFoundError(f"Missing log for {name}: {p}")
        ep, tr, auc = read_log(p)
        curves[name] = (ep, tr, auc)

    if len(curves) == 0:
        raise RuntimeError("No logs found to plot (all missing or skipped).")

    title_suffix = f" {args.title_suffix}".rstrip()

    # ===== 1) Train Objective =====
    plt.figure()
    for name, (ep, tr, auc) in curves.items():
        plt.plot(ep, tr, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Train Objective")
    plt.title(f"Train Objective vs Epoch ({args.dataset}){title_suffix}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out1 = os.path.join(args.img_dir, f"{args.dataset}_trainobj_compare.png")
    plt.savefig(out1, dpi=args.dpi)

    # ===== 2) Test AUC =====
    plt.figure()
    for name, (ep, tr, auc) in curves.items():
        if args.auc_as_percent:
            plt.plot(ep, [a * 100.0 for a in auc], label=name)
        else:
            plt.plot(ep, auc, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Test AUC (%)" if args.auc_as_percent else "Test AUC")
    plt.title(f"Test AUC vs Epoch ({args.dataset}){title_suffix}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out2 = os.path.join(args.img_dir, f"{args.dataset}_testauc_compare.png")
    plt.savefig(out2, dpi=args.dpi)

    print("Saved:")
    print(" ", out1)
    print(" ", out2)


if __name__ == "__main__":
    main()

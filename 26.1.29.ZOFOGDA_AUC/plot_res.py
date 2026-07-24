import os
import csv
import numpy as np
import matplotlib.pyplot as plt


def read_log(csv_path):
    """
    Robust CSV reader:
    - skips rows where epoch is not an int (e.g., '#cfg' line)
    - returns epochs (np.array), train_loss (np.array), test_auc (np.array)
    """
    epochs = []
    train_obj = []
    test_auc = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)

        required = {"epoch", "train_loss", "test_auc"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"[{csv_path}] CSV header missing required columns. "
                f"Need {required}, got {reader.fieldnames}"
            )

        for row in reader:
            e = (row.get("epoch") or "").strip()
            if not e.isdigit():
                continue

            try:
                epochs.append(int(e))
                train_obj.append(float(row["train_loss"]))
                test_auc.append(float(row["test_auc"]))
            except Exception:
                continue

    if len(epochs) == 0:
        raise ValueError(f"[{csv_path}] No valid epoch rows found (after skipping non-numeric epochs).")

    return np.array(epochs, dtype=int), np.array(train_obj, dtype=float), np.array(test_auc, dtype=float)


def align_runs_by_epoch(runs):
    """
    runs: list of (epochs, train_obj, test_auc)
    Align by common epoch intersection across seeds.
    """
    epoch_sets = [set(r[0].tolist()) for r in runs]
    common = set.intersection(*epoch_sets)
    if len(common) == 0:
        raise ValueError("No common epochs across seeds. Check logs or alignment strategy.")

    common_epochs = np.array(sorted(common), dtype=int)

    train_rows = []
    auc_rows = []
    for ep, tr, auc in runs:
        idx = {e: i for i, e in enumerate(ep.tolist())}
        train_rows.append(np.array([tr[idx[e]] for e in common_epochs], dtype=float))
        auc_rows.append(np.array([auc[idx[e]] for e in common_epochs], dtype=float))

    train_mat = np.stack(train_rows, axis=0)  # [n_seeds, T]
    auc_mat = np.stack(auc_rows, axis=0)      # [n_seeds, T]
    return common_epochs, train_mat, auc_mat


def band_stat(y_mat, mode="std"):
    """
    y_mat: [n_seeds, T]
    mode:
      - "std": mean ± std
      - "sem": mean ± std/sqrt(n)
      - "var": mean ± var
      - "minmax": [min, max] band
    """
    mean = y_mat.mean(axis=0)

    if mode == "std":
        band = y_mat.std(axis=0, ddof=0)
        lo, hi = mean - band, mean + band
    elif mode == "sem":
        std = y_mat.std(axis=0, ddof=0)
        band = std / np.sqrt(max(1, y_mat.shape[0]))
        lo, hi = mean - band, mean + band
    elif mode == "var":
        var = y_mat.var(axis=0, ddof=0)
        lo, hi = mean - var, mean + var
    elif mode == "minmax":
        lo, hi = y_mat.min(axis=0), y_mat.max(axis=0)
    else:
        raise ValueError(f"Unknown band mode: {mode}")

    return mean, lo, hi


def plot_mean_with_band(x, y_mean, y_lo, y_hi, label, color=None, alpha=0.2, linewidth=10.5):
    """
    color=None -> let matplotlib auto-color in plot; we then reuse same color for fill.
    """
    line, = plt.plot(x, y_mean, label=label, color=color, linewidth=linewidth)
    c = line.get_color()
    plt.fill_between(x, y_lo, y_hi, color=c, alpha=alpha)


def main():
    # ====== CONFIG ======
    data_name = "ijcnn1"
    band_mode = "std"

    out_dir = f"./img/{data_name}"
    os.makedirs(out_dir, exist_ok=True)

    # ====== Plot style knobs (论文风格可微调) ======
    LABEL_FONTSIZE = 156     # 横纵坐标字体
    TICK_FONTSIZE = 153      # 刻度字体
    TITLE_FONTSIZE = 154     # 标题字体
    LEGEND_FONTSIZE = 152    # 图例字体
    GRID_ALPHA = 0.18       # 网格线更浅
    GRID_LW = 0.6           # 网格线更细

    # ====== log files (3 seeds) ======
    ttsgda_csvs = [
        f"./result_plot/{data_name}/ttsgda_auc_ijcnn1_epochs500_seed14_batch128_lrx0.01_lry0.0001_log.csv",
        f"./result_plot/{data_name}/ttsgda_auc_ijcnn1_epochs500_seed42_batch128_lrx0.01_lry0.0001_log.csv",
        f"./result_plot/{data_name}/ttsgda_auc_ijcnn1_epochs500_seed81_batch128_lrx0.01_lry0.0001_log.csv",
    ]
    zofogda_csvs = [
        f"./result_plot/{data_name}/zofoGDA_auc_ijcnn1_epochs500_seed14_batch128_gamma0.01_lam0.01_delta0.001q5_log.csv",
        f"./result_plot/{data_name}/zofoGDA_auc_ijcnn1_epochs500_seed42_batch128_gamma0.01_lam0.01_delta0.001q5_log.csv",
        f"./result_plot/{data_name}/zofoGDA_auc_ijcnn1_epochs500_seed81_batch128_gamma0.01_lam0.01_delta0.001q5_log.csv",
    ]
    ttgda_csvs = [
        f"./result_plot/{data_name}/ttgda_auc_ijcnn1_epochs500_seed14_batch128_lrx0.08_lry0.0008_log.csv",
        f"./result_plot/{data_name}/ttgda_auc_ijcnn1_epochs500_seed42_batch128_lrx0.08_lry0.0008_log.csv",
        f"./result_plot/{data_name}/ttgda_auc_ijcnn1_epochs500_seed81_batch128_lrx0.08_lry0.0008_log.csv",
    ]

    def load_method(csv_list):
        runs = []
        for p in csv_list:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing log: {p}")
            runs.append(read_log(p))
        return align_runs_by_epoch(runs)

    # ====== Load & align ======
    ep_t, obj_t_mat, auc_t_mat = load_method(ttsgda_csvs)
    ep_z, obj_z_mat, auc_z_mat = load_method(zofogda_csvs)
    ep_g, obj_g_mat, auc_g_mat = load_method(ttgda_csvs)

    # =========================
    # 1) Train Objectivet
    # =========================
    plt.figure(figsize=(40.0, 39.0))

    obj_t_mean, obj_t_lo, obj_t_hi = band_stat(obj_t_mat, mode=band_mode)
    obj_g_mean, obj_g_lo, obj_g_hi = band_stat(obj_g_mat, mode=band_mode)
    obj_z_mean, obj_z_lo, obj_z_hi = band_stat(obj_z_mat, mode=band_mode)

    # 图例顺序：TTSGDA, TTGDA, ZOFOGDA（zofogda放最后）
    plot_mean_with_band(ep_t, obj_t_mean, obj_t_lo, obj_t_hi, label="TTSGDA")
    plot_mean_with_band(ep_g, obj_g_mean, obj_g_lo, obj_g_hi, label="TTGDA")
    plot_mean_with_band(ep_z, obj_z_mean, obj_z_lo, obj_z_hi, label="ZOFOGDA")

    plt.xlabel("Epoch", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Train Objective", fontsize=LABEL_FONTSIZE)
    plt.title(f"Train Objective vs Epoch ({band_mode} band over seeds)", fontsize=TITLE_FONTSIZE)

    plt.grid(True, alpha=GRID_ALPHA, linewidth=GRID_LW)
    plt.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    plt.legend(fontsize=LEGEND_FONTSIZE)

    out1 = os.path.join(out_dir, f"{data_name}_trainobj_compare.png")
    plt.tight_layout()
    plt.savefig(out1, dpi=800, bbox_inches="tight")
    plt.savefig(out1.replace(".png", ".pdf"), bbox_inches="tight")

    # =========================
    # 2) Test AUC (0~1)
    # =========================
    plt.figure(figsize=(40.0, 39.0))

    auc_t_mean, auc_t_lo, auc_t_hi = band_stat(auc_t_mat, mode=band_mode)
    auc_g_mean, auc_g_lo, auc_g_hi = band_stat(auc_g_mat, mode=band_mode)
    auc_z_mean, auc_z_lo, auc_z_hi = band_stat(auc_z_mat, mode=band_mode)

    # 不再乘100，直接画0~1的小数
    plot_mean_with_band(ep_t, auc_t_mean, auc_t_lo, auc_t_hi, label="TTSGDA")
    plot_mean_with_band(ep_g, auc_g_mean, auc_g_lo, auc_g_hi, label="TTGDA")
    plot_mean_with_band(ep_z, auc_z_mean, auc_z_lo, auc_z_hi, label="ZOFOGDA")

    plt.xlabel("Epoch", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Test AUC", fontsize=LABEL_FONTSIZE)
    plt.title(f"Test AUC vs Epoch", fontsize=TITLE_FONTSIZE)

    plt.grid(True, alpha=GRID_ALPHA, linewidth=GRID_LW)
    plt.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    plt.legend(fontsize=LEGEND_FONTSIZE)

    out2 = os.path.join(out_dir, f"{data_name}_testauc_compare.png")
    plt.tight_layout()
    plt.savefig(out2, dpi=800, bbox_inches="tight")
    plt.savefig(out2.replace(".png", ".pdf"), bbox_inches="tight")

    print("Saved:")
    print(" ", out1)
    print(" ", out2)


if __name__ == "__main__":
    main()

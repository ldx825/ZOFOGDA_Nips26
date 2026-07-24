import csv
import matplotlib.pyplot as plt


def read_log(csv_path):
    epochs = []
    train_obj = []
    test_acc = []
    y_min = []
    y_max = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_obj.append(float(row["train_obj"]))
            test_acc.append(float(row["test_acc"]))
            y_min.append(float(row["y_min"]))
            y_max.append(float(row["y_max"]))

    return epochs, train_obj, test_acc, y_min, y_max


def main():
    # ====== log files ======
    data_name="svmguide1"
    ttsgda_csv = "./result/ttsgda_"+data_name+"_log.csv"
    zofogda_csv = "./result/zofoGDA_"+data_name+"_log.csv"
    # NEW: TTGDA log（把文件名换成你实际保存的）
    ttgda_csv = "./result/ttgda_"+data_name+"_log.csv"

    # ====== read logs ======
    ep_t, obj_t, acc_t, y_min_t, y_max_t = read_log(ttsgda_csv)
    ep_z, obj_z, acc_z, y_min_z, y_max_z = read_log(zofogda_csv)
    ep_g, obj_g, acc_g, y_min_g, y_max_g = read_log(ttgda_csv)

    # ====== 1) Train Objective ======
    plt.figure()
    plt.plot(ep_t, obj_t, label="TTSGDA")
    plt.plot(ep_z, obj_z, label="ZOFOGDA")
    plt.plot(ep_g, obj_g, label="TTGDA")
    plt.xlabel("Epoch")
    plt.ylabel("Train Objective")
    plt.title("Train Objective vs Epoch")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("./img/"+data_name+"_trainobj_compare.png", dpi=200)

    # ====== 2) Test Accuracy ======
    plt.figure()
    plt.plot(ep_t, [a * 100 for a in acc_t], label="TTSGDA")
    plt.plot(ep_z, [a * 100 for a in acc_z], label="ZOFOGDA")
    plt.plot(ep_g, [a * 100 for a in acc_g], label="TTGDA")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Test Accuracy vs Epoch")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("./img/"+data_name+"_testacc_compare.png", dpi=200)

    # ====== 3) y range ======
    plt.figure()
    plt.plot(ep_t, y_min_t, "--", label="y_min (TTSGDA)")
    plt.plot(ep_t, y_max_t, "--", label="y_max (TTSGDA)")

    plt.plot(ep_z, y_min_z, "-", label="y_min (ZOFOGDA)")
    plt.plot(ep_z, y_max_z, "-", label="y_max (ZOFOGDA)")

    plt.plot(ep_g, y_min_g, ":", label="y_min (TTGDA)")
    plt.plot(ep_g, y_max_g, ":", label="y_max (TTGDA)")

    plt.xlabel("Epoch")
    plt.ylabel("y value")
    plt.title("y[min, max] vs Epoch")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("./img/"+data_name+"_y_range_compare.png", dpi=200)

    print("Saved:")
    print("  ./img/"+data_name+"_trainobj_compare.png")
    print("  ./img/"+data_name+"_testacc_compare.png")
    print("  ./img/"+data_name+"_y_range_compare.png")


if __name__ == "__main__":
    main()

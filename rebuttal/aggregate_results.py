#!/usr/bin/env python3
"""Refresh the rebuttal Excel result workbook and query-budget plots."""
from __future__ import annotations
import argparse
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

ROOT = Path(__file__).resolve().parents[1]
REB = ROOT / "rebuttal"
REGISTRY = REB / "tables" / "experiment_registry.csv"
WORKBOOK = REB / "tables" / "rebuttal_experiment_results.xlsx"
SIMPLE_WORKBOOK = REB / "tables" / "rebuttal_results_simple.xlsx"
SIMPLE_PREVIOUS_WORKBOOK = REB / "tables" / "rebuttal_results_simple_previous.xlsx"
PLOTS = REB / "plots"


def collect():
    registry = pd.read_csv(REGISTRY)
    result_rows = []
    curves = []
    status_map = {}
    for _, reg in registry.iterrows():
        out = ROOT / str(reg["output_dir"])
        summary_path = out / "summary.json"
        status = "pending"
        summary = {}
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                status = summary.get("status", "unknown")
            except Exception as exc:
                status = "broken"
                summary = {"error": repr(exc)}
        status_map[reg["run_id"]] = status
        row = reg.to_dict()
        row["status"] = status
        for k, v in summary.items():
            if k not in row or k in {
                "final_test_metric", "best_test_metric", "final_train_objective", "global_steps",
                "cum_train_forward_queries", "cum_x_function_queries", "cum_train_backward_calls",
                "cum_train_sample_forwards", "train_wall_time_sec", "eval_wall_time_sec",
                "n_train", "n_test", "feature_dim", "parameter_dim", "completed_at", "error"
            }:
                row[k] = v
        result_rows.append(row)
        metrics = out / "metrics.csv"
        if metrics.exists():
            try:
                df = pd.read_csv(metrics)
                if not df.empty:
                    df["experiment_groups"] = reg["experiment_groups"]
                    df["notes"] = reg.get("notes", "")
                    df["batch_size"] = reg["batch_size"]
                    df["hidden"] = reg["hidden"]
                    df["c1"] = reg["c1"]
                    df["c2"] = reg["c2"]
                    df["direction_distribution"] = reg["direction_distribution"]
                    curves.append(df)
            except Exception:
                pass
    registry["status"] = registry["run_id"].map(status_map)
    raw = pd.DataFrame(result_rows)
    curve_df = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    return registry, raw, curve_df


def explode_groups(df):
    if df.empty:
        return df
    x = df.copy()
    x["experiment_group"] = x["experiment_groups"].fillna("").str.split(",")
    return x.explode("experiment_group")


def summarize(raw):
    done = raw[raw["status"] == "completed"].copy() if not raw.empty else raw
    if done.empty:
        return pd.DataFrame()
    done = explode_groups(done)
    keys = [
        "experiment_group", "task", "dataset", "method", "estimator", "M", "delta",
        "x_momentum", "y_momentum", "direction_distribution", "batch_size", "hidden",
        "c1", "c2", "lr_x", "lr_y"
    ]
    values = [
        "final_test_metric", "best_test_metric", "global_steps", "cum_train_forward_queries",
        "cum_train_backward_calls", "cum_train_sample_forwards", "train_wall_time_sec"
    ]
    agg = done.groupby(keys, dropna=False)[values].agg(["mean", "std", "count"]).reset_index()
    agg.columns = ["_".join(str(y) for y in x if str(y)) if isinstance(x, tuple) else str(x) for x in agg.columns]
    return agg.sort_values(["experiment_group", "task", "dataset"])


def status_dashboard(registry):
    return (registry.groupby(["phase", "assigned_gpu", "status"], dropna=False)
            .size().rename("runs").reset_index().sort_values(["phase", "assigned_gpu", "status"]))


def momentum_label(row):
    x = bool(row.get("x_momentum", True)); y = bool(row.get("y_momentum", True))
    return "full" if x and y else "no_x" if (not x and y) else "no_y" if (x and not y) else "no_both"


def curve_label(group, row):
    if group == "main_comparison": return str(row["method"])
    if group == "M_sweep": return f"M={int(row['M'])}"
    if group == "delta_sweep": return f"delta={row['delta']:g}"
    if group == "momentum": return momentum_label(row)
    if group == "perturbation_form": return str(row["estimator"])
    if group == "direction_ablation": return str(row["direction_distribution"])
    if group == "dimension": return f"hidden={int(row['hidden'])}"
    return str(row.get("method", "curve"))


def make_plots(curves):
    if curves.empty:
        return
    PLOTS.mkdir(parents=True, exist_ok=True)
    groups = ["main_comparison", "M_sweep", "delta_sweep", "momentum", "perturbation_form", "direction_ablation", "dimension"]
    x = explode_groups(curves)
    for group in groups:
        sub = x[x["experiment_group"] == group].copy()
        if sub.empty: continue
        sub["label"] = sub.apply(lambda r: curve_label(group, r), axis=1)
        for (task, dataset), ds in sub.groupby(["task", "dataset"]):
            plt.figure(figsize=(7.2, 4.8))
            for label, c in ds.groupby("label"):
                g = c.groupby("target_query")["test_metric"].agg(["mean", "std"]).reset_index().sort_values("target_query")
                xx = g["target_query"].to_numpy(); yy = g["mean"].to_numpy(); ss = g["std"].fillna(0).to_numpy()
                plt.plot(xx, yy, label=label, linewidth=1.8)
                plt.fill_between(xx, yy-ss, yy+ss, alpha=0.16)
            plt.xlabel("Training forward queries")
            plt.ylabel("Test AUC" if task == "auc" else "Test accuracy")
            plt.title(f"{group}: {task}/{dataset}")
            plt.grid(alpha=0.25); plt.legend(fontsize=8); plt.tight_layout()
            out = PLOTS / group
            out.mkdir(parents=True, exist_ok=True)
            plt.savefig(out / f"{task}_{dataset}.png", dpi=180)
            plt.close()


def write_excel(registry, raw, curves, summary):
    WORKBOOK.parent.mkdir(parents=True, exist_ok=True)
    group_sheets = {
        "Main Comparison": "main_comparison", "M Sweep": "M_sweep", "Delta Sweep": "delta_sweep",
        "Momentum": "momentum", "Perturbation": "perturbation_form", "Direction": "direction_ablation",
        "Coeff Batch": "coefficient_batch", "Dimension": "dimension",
        "M Delta": "M_delta_interaction", "M Momentum": "M_momentum_interaction", "FO LR Sweep": "fo_lr_sweep",
    }
    with pd.ExcelWriter(WORKBOOK, engine="openpyxl") as writer:
        pd.DataFrame({
            "说明": [
                "本工作簿由 rebuttal/aggregate_results.py 自动生成，请不要手工改 Raw Results。",
                "Run Registry：完整实验表；Status Dashboard：阶段/GPU/状态统计。",
                "Raw Results：每个 run 的最终结果；Raw Curves：每个 query checkpoint 的原始曲线。",
                "Summary 和各实验分页：三 seed 的 mean/std/count 汇总。",
                "主公平口径为 training forward queries；同时保留 backward、sample-forward 和 wall-clock。",
                "运行 status：pending/running/completed/failed。执行 refresh_results.sh 可随时刷新。",
            ]
        }).to_excel(writer, sheet_name="README", index=False)
        status_dashboard(registry).to_excel(writer, sheet_name="Status Dashboard", index=False)
        registry.to_excel(writer, sheet_name="Run Registry", index=False)
        raw.to_excel(writer, sheet_name="Raw Results", index=False)
        (curves if not curves.empty else pd.DataFrame(columns=["no_results_yet"])).to_excel(writer, sheet_name="Raw Curves", index=False)
        (summary if not summary.empty else pd.DataFrame(columns=["no_completed_results_yet"])).to_excel(writer, sheet_name="Summary", index=False)
        for sheet, group in group_sheets.items():
            part = summary[summary["experiment_group"] == group] if not summary.empty else pd.DataFrame()
            (part if not part.empty else pd.DataFrame(columns=["no_completed_results_yet"])).to_excel(writer, sheet_name=sheet[:31], index=False)
    wb = load_workbook(WORKBOOK)
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)
    status_colors = {"completed": "C6EFCE", "running": "FFF2CC", "failed": "FFC7CE", "pending": "D9EAF7"}
    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        if ws.max_row >= 1 and ws.max_column >= 1:
            ws.auto_filter.ref = ws.dimensions
        for cell in ws[1]:
            cell.fill = header_fill; cell.font = header_font; cell.alignment = Alignment(horizontal="center")
        for col in ws.columns:
            vals = [str(c.value) if c.value is not None else "" for c in list(col)[:200]]
            width = min(42, max(10, max((len(v) for v in vals), default=10) + 2))
            ws.column_dimensions[col[0].column_letter].width = width
        headers = {c.value: c.column for c in ws[1]}
        if "status" in headers:
            col = headers["status"]
            for row in range(2, ws.max_row + 1):
                cell = ws.cell(row, col); color = status_colors.get(str(cell.value))
                if color: cell.fill = PatternFill("solid", fgColor=color)
    wb.save(WORKBOOK)



def method_name(method):
    return {"zofo": "ZOFOSGDA", "ttgda": "TTGDA", "ttsgda": "TTSGDA"}.get(str(method), str(method))



def write_previous_simple_workbook(simple):
    """Keep the original one-sheet six-column layout requested by the user."""
    flat = simple.copy()
    if not flat.empty:
        flat = flat.sort_values(["所做任务", "数据集", "消融变量", "方法", "对应的值"])
    with pd.ExcelWriter(SIMPLE_PREVIOUS_WORKBOOK, engine="openpyxl") as writer:
        flat.to_excel(writer, sheet_name="实验结果简表", index=False)
    wb = load_workbook(SIMPLE_PREVIOUS_WORKBOOK)
    ws = wb["实验结果简表"]
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    header_fill = PatternFill("solid", fgColor="1F4E78")
    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = Font(color="FFFFFF", bold=True)
        cell.alignment = Alignment(horizontal="center")
    for col, width in {"A": 20, "B": 16, "C": 24, "D": 32, "E": 16, "F": 24}.items():
        ws.column_dimensions[col].width = width
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="center")
        text = str(row[5].value)
        if text.startswith("待运行"):
            row[5].fill = PatternFill("solid", fgColor="D9EAF7")
        elif text.startswith("运行中") or text.startswith("当前"):
            row[5].fill = PatternFill("solid", fgColor="FFF2CC")
        else:
            row[5].fill = PatternFill("solid", fgColor="C6EFCE")
    wb.save(SIMPLE_PREVIOUS_WORKBOOK)

def simple_result_workbook(raw):
    """Write the six-column, teacher-facing result table."""
    columns = ["数据集", "方法", "消融变量", "对应的值", "所做任务", "结果"]
    if raw.empty:
        simple = pd.DataFrame(columns=columns)
    else:
        x = explode_groups(raw)
        specs = {
            "main_comparison": ("方法对比", ["method"]),
            "M_sweep": ("M", ["method", "M"]),
            "delta_sweep": ("扰动半径 delta", ["method", "delta"]),
            "momentum": ("动量", ["method", "x_momentum", "y_momentum"]),
            "perturbation_form": ("零阶扰动形式", ["method", "estimator"]),
            "direction_ablation": ("扰动方向分布", ["method", "direction_distribution"]),
            "coefficient_batch": ("c1=c2 / batch size", ["method", "c1", "c2", "batch_size"]),
            "dimension": ("隐藏层维度", ["method", "hidden"]),
            "M_delta_interaction": ("M / delta", ["method", "M", "delta"]),
            "M_momentum_interaction": ("M / 动量", ["method", "M", "x_momentum", "y_momentum"]),
            "fo_lr_sweep": ("一阶学习率 lr_x / lr_y", ["method", "lr_x", "lr_y"]),
        }
        rows = []
        for group, (variable, variant_keys) in specs.items():
            part = x[x["experiment_group"] == group].copy()
            if part.empty:
                continue
            keys = ["task", "dataset"] + variant_keys
            for key_values, g in part.groupby(keys, dropna=False, sort=True):
                if not isinstance(key_values, tuple):
                    key_values = (key_values,)
                values = dict(zip(keys, key_values))
                method = values.get("method", "zofo")
                if group == "main_comparison":
                    value = "默认设置"
                elif group == "M_sweep":
                    value = f"M={int(values['M'])}"
                elif group == "delta_sweep":
                    value = f"delta={float(values['delta']):g}"
                elif group == "momentum":
                    value = momentum_label(values)
                elif group == "perturbation_form":
                    value = str(values["estimator"])
                elif group == "direction_ablation":
                    value = str(values["direction_distribution"])
                elif group == "coefficient_batch":
                    value = f"c={float(values['c1']):g}, batch={int(values['batch_size'])}"
                elif group == "dimension":
                    value = f"hidden={int(values['hidden'])}"
                elif group == "M_delta_interaction":
                    value = f"M={int(values['M'])}, delta={float(values['delta']):g}"
                elif group == "M_momentum_interaction":
                    value = f"M={int(values['M'])}, {momentum_label(values)}"
                else:
                    value = f"lr_x={float(values['lr_x']):g}, lr_y={float(values['lr_y']):g}"

                completed = pd.to_numeric(
                    g.loc[g["status"] == "completed", "final_test_metric"], errors="coerce"
                ).dropna()
                n_done, n_total = len(completed), len(g)
                if n_done == n_total and n_done > 1:
                    result = f"{completed.mean():.4f} ± {completed.std(ddof=1):.4f}"
                elif n_done == n_total == 1:
                    result = f"{completed.iloc[0]:.4f}"
                elif n_done:
                    result = f"当前 {completed.mean():.4f}（{n_done}/{n_total}）"
                else:
                    running = int((g["status"] == "running").sum())
                    result = f"运行中（0/{n_total}）" if running else f"待运行（0/{n_total}）"
                rows.append({
                    "数据集": values["dataset"],
                    "方法": method_name(method),
                    "消融变量": variable,
                    "对应的值": value,
                    "所做任务": "AUC 最大化" if values["task"] == "auc" else "稳健分类",
                    "结果": result,
                })
        simple = pd.DataFrame(rows, columns=columns)

    write_previous_simple_workbook(simple)

    method_order = {"ZOFOSGDA": 0, "TTGDA": 1, "TTSGDA": 2}
    methods = list(method_order)
    comparison = simple[simple["消融变量"] == "方法对比"].copy()
    ablations = simple[simple["消融变量"] != "方法对比"].copy()
    comparison["_method_order"] = comparison["方法"].map(method_order).fillna(99)
    comparison = comparison.sort_values(["所做任务", "数据集", "_method_order"]).drop(columns="_method_order")

    # For every ablation condition, place the condition-specific method and the
    # other two methods' equal-query main baselines in one consecutive triplet.
    baseline = {
        (r["所做任务"], r["数据集"], r["方法"]): r["结果"]
        for _, r in comparison.iterrows()
    }
    triplet_rows = []
    condition_keys = ["所做任务", "数据集", "消融变量", "对应的值"]
    for condition, g in ablations.groupby(condition_keys, dropna=False, sort=True):
        task_name, dataset, variable, value = condition
        actual = {r["方法"]: r for _, r in g.iterrows()}
        for method in methods:
            if method in actual:
                result = actual[method]["结果"]
                shown_value = value
            else:
                result = baseline.get((task_name, dataset, method), "无对应基线")
                shown_value = "同 query 基线"
            triplet_rows.append({
                "数据集": dataset,
                "方法": method,
                "消融变量": variable,
                "对应的值": shown_value,
                "所做任务": task_name,
                "结果": result,
            })
    ablation_triplets = pd.DataFrame(triplet_rows, columns=columns)
    grouped = pd.concat([comparison, ablation_triplets], ignore_index=True)

    with pd.ExcelWriter(SIMPLE_WORKBOOK, engine="openpyxl") as writer:
        grouped.to_excel(writer, sheet_name="三方法成组", index=False)
        comparison.to_excel(writer, sheet_name="仅主对比", index=False)
    wb = load_workbook(SIMPLE_WORKBOOK)
    header_fill = PatternFill("solid", fgColor="1F4E78")
    widths = {"A": 20, "B": 16, "C": 24, "D": 32, "E": 16, "F": 24}
    top = Side(style="medium", color="5B9BD5")
    cluster_fills = [PatternFill("solid", fgColor="EAF2F8"), PatternFill("solid", fgColor="F7F9F9")]
    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = Font(color="FFFFFF", bold=True)
            cell.alignment = Alignment(horizontal="center")
        for col, width in widths.items():
            ws.column_dimensions[col].width = width
        for start in range(2, ws.max_row + 1, 3):
            fill = cluster_fills[((start - 2) // 3) % 2]
            for r in range(start, min(start + 3, ws.max_row + 1)):
                for c in range(1, 6):
                    ws.cell(r, c).fill = fill
                for c in range(1, 7):
                    cell = ws.cell(r, c)
                    cell.alignment = Alignment(vertical="center")
                    if r == start:
                        cell.border = Border(top=top)
                result_cell = ws.cell(r, 6)
                text = str(result_cell.value)
                if text.startswith("待运行"):
                    result_cell.fill = PatternFill("solid", fgColor="D9EAF7")
                elif text.startswith("运行中") or text.startswith("当前"):
                    result_cell.fill = PatternFill("solid", fgColor="FFF2CC")
                elif text == "无对应基线":
                    result_cell.fill = PatternFill("solid", fgColor="E7E6E6")
                else:
                    result_cell.fill = PatternFill("solid", fgColor="C6EFCE")
    wb.save(SIMPLE_WORKBOOK)

def refresh():
    registry, raw, curves = collect()
    summary = summarize(raw)
    write_excel(registry, raw, curves, summary)
    simple_result_workbook(raw)
    make_plots(curves)
    counts = registry["status"].value_counts().to_dict()
    print(time.strftime("%F %T"), counts, "->", WORKBOOK, SIMPLE_WORKBOOK, SIMPLE_PREVIOUS_WORKBOOK, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--interval", type=int, default=600)
    args = ap.parse_args()
    while True:
        try: refresh()
        except Exception as exc: print("refresh failed:", repr(exc), flush=True)
        if not args.watch: break
        time.sleep(args.interval)

if __name__ == "__main__": main()

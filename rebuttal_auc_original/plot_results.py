#!/usr/bin/env python3
"""Paper-style AUC curves from the original-script experiment outputs."""
import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import manage

COLORS = {"TTGDA": "#2ca02c", "TTSGDA": "#2345ff", "ZOFOSGDA": "#ff2d2d",
          "M=1": "#1f77b4", "M=5": "#ff7f0e", "M=10": "#2ca02c", "M=15": "#d62728",
          "M=floor(sqrt(d))": "#9467bd",
          "delta=0.0001": "#1f77b4", "delta=0.001": "#ff2d2d", "delta=0.01": "#2ca02c",
          "full": "#ff2d2d", "no-x": "#1f77b4", "no-y": "#2ca02c", "no-both": "#9467bd"}


def read_numeric(path):
    rows=[]
    with open(path,newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("epoch") or "").isdigit(): rows.append(row)
    return rows


def manifest_lookup():
    runs=manage.build()
    return {(r["dataset"],r["method"],r["experiment"],r["value"],r["seed"]):r for r in runs}


def resolve_key(lookup,dataset,method,experiment,value,seed):
    if experiment=="M" and str(value)=="sqrt_d": value=manage.SQRT_D_M[dataset]
    elif experiment=="M" and str(value)=="5": experiment,value="baseline","paper"
    if experiment=="delta" and str(value) in {"0.001","1e-3"}: experiment,value="baseline","paper"
    if experiment=="momentum" and value=="full": experiment,value="baseline","paper"
    return lookup[(dataset,method,experiment,str(value),seed)]


def run_curve(run,xaxis,allow_paper=True):
    summary_path=manage.RUNS_DIR/run["run_id"] / "summary.json"
    summary=None
    if summary_path.exists():
        summary=json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status")!="completed": summary=None
    if summary and summary.get("gpu_exclusive") is True:
        log_path=Path(summary["csv_path"])
    else:
        return None
    rows=read_numeric(log_path)
    epochs=np.array([int(r["epoch"]) for r in rows],dtype=float)
    auc=np.array([float(r["test_auc"]) for r in rows],dtype=float)*100.0
    if xaxis=="epoch": x=epochs
    elif xaxis=="query":
        mult=2*run["q"]+1 if run["method"]=="ZOFOSGDA" else 1
        x=epochs*run["n_train"]*mult
    else:
        if not summary or not summary.get("epoch_times_path"): return None
        times=read_numeric(summary["epoch_times_path"])
        if len(times)!=len(rows): return None
        x=np.array([float(r["time_ms"]) for r in times])
    return x,auc


def specs(comparison):
    if comparison=="baseline":
        return [(m,m,"baseline","paper") for m in ["TTGDA","TTSGDA","ZOFOSGDA"]]
    if comparison=="M":
        fixed=[(f"M={v}","ZOFOSGDA","M",v) for v in [1,5,10,15]]
        return fixed+[("M=floor(sqrt(d))","ZOFOSGDA","M","sqrt_d")]
    if comparison=="delta":
        return [(f"delta={v}","ZOFOSGDA","delta",v) for v in [0.0001,0.001,0.01]]
    experiment = "fixed_momentum" if comparison == "fixed_momentum" else "momentum"
    return [(label,"ZOFOSGDA",experiment,value) for label,value in
            [("full","full"),("no-x","no_x"),("no-y","no_y"),("no-both","no_both")]]


def aggregate(curves,xaxis):
    if len(curves)<2: return None
    if xaxis=="epoch":
        end=min(len(x) for x,_ in curves)
        grid=curves[0][0][:end]
        mat=np.stack([y[:end] for _,y in curves])
    else:
        lo=max(x[0] for x,_ in curves); hi=min(x[-1] for x,_ in curves)
        if hi<=lo: return None
        grid=np.linspace(lo,hi,500)
        mat=np.stack([np.interp(grid,x,y) for x,y in curves])
    return grid,mat.mean(0),mat.std(0,ddof=0)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--datasets",default=",".join(manage.DATASETS),help="comma-separated")
    p.add_argument("--comparison",choices=["baseline","M","delta","momentum","fixed_momentum"],default="baseline")
    p.add_argument("--xaxis",choices=["epoch","time","query"],default="epoch")
    p.add_argument("--output",default="")
    args=p.parse_args()
    datasets=[x.strip() for x in args.datasets.split(",") if x.strip()]
    lookup=manifest_lookup()
    n=len(datasets); cols=min(3,n); rows=(n+cols-1)//cols
    fig,axes=plt.subplots(rows,cols,figsize=(5.0*cols,4.1*rows),squeeze=False)
    missing=[]
    for ax,d in zip(axes.flat,datasets):
        for label,method,experiment,value in specs(args.comparison):
            curves=[]
            for seed in manage.SEEDS:
                run=resolve_key(lookup,d,method,experiment,value,seed)
                curve=run_curve(run,args.xaxis)
                if curve is not None: curves.append(curve)
            agg=aggregate(curves,args.xaxis)
            if agg is None:
                missing.append(f"{d}:{label}"); continue
            x,mean,std=agg; color=COLORS.get(label)
            ax.plot(x,mean,label=label,color=color,linewidth=2.0)
            ax.fill_between(x,mean-std,mean+std,color=color,alpha=0.18,linewidth=0)
        ax.set_title(d.capitalize(),fontsize=13)
        ax.set_ylabel("Test AUC (%)")
        ax.set_xlabel({"epoch":"Epoch","time":"Time (ms)","query":"Sample-function queries"}[args.xaxis])
        ax.grid(True,alpha=.22,linewidth=.6)
        ax.legend(frameon=True,fontsize=9)
        if args.xaxis=="query": ax.ticklabel_format(axis="x",style="sci",scilimits=(0,0))
    for ax in axes.flat[n:]: ax.axis("off")
    fig.tight_layout()
    out=Path(args.output) if args.output else HERE/"plots"/f"{args.comparison}_{args.xaxis}.pdf"
    out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"),dpi=300,bbox_inches="tight")
    print(f"saved {out} and {out.with_suffix('.png')}")
    if missing: print("waiting for complete 3-seed data:",", ".join(missing))

if __name__=="__main__": main()

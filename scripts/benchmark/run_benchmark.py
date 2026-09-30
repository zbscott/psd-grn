#!/usr/bin/env python3
"""Run PSD-GRN benchmark tasks with the locked article hyperparameters.

Expected data layout under --data-root:
    <dataset>/ExpressionData500.csv or ExpressionData1000.csv
    <dataset>/PseudoTime.csv
    <dataset>/gold_augmented.csv
"""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

EDGE_DATASETS=["hESC","hHep","mDC","mESC","mHSC-E","mHSC-GM","mHSC-L"]
SIGNED_DATASETS=["hESC","hHep","mDC","mESC","mHSC-E","mHSC-GM"]


def run(cmd, log):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w',encoding='utf-8') as f:
        p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
    if p.returncode:
        raise SystemExit(f"Failed ({p.returncode}): {' '.join(map(str,cmd))}\nSee {log}")


def validate_dataset(data_root: Path, dataset: str, expression_file: str):
    base=data_root/dataset
    missing=[str(base/x) for x in [expression_file,"PseudoTime.csv","gold_augmented.csv"] if not (base/x).exists()]
    if missing:
        raise FileNotFoundError("Missing benchmark input(s):\n"+"\n".join(missing))


def common(root,dataroot,outroot,dataset,expr,script):
    return [sys.executable,"-u",str(root/"src"/script),
            "--dataset",dataset,"--expression_file",expr,
            "--data_root",str(dataroot),"--output_root",str(outroot),
            "--K","1","--q","0.1","--hidden","64","--bins","5",
            "--lr","0.001","--weight_decay","0.05","--dropout","0.5",
            "--epochs","1000","--patience","20","--checkpoint","25",
            "--train_ratio","0.8","--val_ratio","0.1","--runs","5",
            "--seed","0","--method","PSD-GRN"]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[2])
    ap.add_argument('--data-root',type=Path,required=True)
    ap.add_argument('--output-root',type=Path,default=None)
    ap.add_argument('--task',choices=['edge','signed','both'],default='both')
    ap.add_argument('--gene-set',choices=['500','1000'],default='1000')
    ap.add_argument('--datasets',nargs='*',default=None,help='Optional dataset subset, e.g. hESC hHep')
    args=ap.parse_args(); root=args.repo_root.resolve(); out=(args.output_root or root).resolve(); data=args.data_root.resolve()
    expr=f"ExpressionData{args.gene_set}.csv"

    if args.task in {'signed','both'}:
        ds_list=args.datasets or SIGNED_DATASETS
        invalid=sorted(set(ds_list)-set(SIGNED_DATASETS))
        if invalid: raise ValueError(f"Invalid signed-task dataset(s): {invalid}; mHSC-L is excluded by protocol.")
        for ds in ds_list:
            validate_dataset(data,ds,expr)
            cmd=common(root,data,out,ds,expr,"train_psdgrn_3c.py")+["--num_classes","3"]
            print('[SIGNED]',ds); run(cmd,out/"benchmark_logs"/f"signed_{ds}_TF{args.gene_set}.log")
    if args.task in {'edge','both'}:
        ds_list=args.datasets or EDGE_DATASETS
        invalid=sorted(set(ds_list)-set(EDGE_DATASETS))
        if invalid: raise ValueError(f"Invalid edge-task dataset(s): {invalid}")
        for ds in ds_list:
            validate_dataset(data,ds,expr)
            cmd=common(root,data,out,ds,expr,"train_psdgrn_2c.py")+["--num_classes","2"]
            print('[EDGE]',ds); run(cmd,out/"benchmark_logs"/f"edge_{ds}_TF{args.gene_set}.log")

if __name__=='__main__': main()

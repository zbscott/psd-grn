#!/usr/bin/env python3
"""Cross-platform runner for formal T=3,4,5,6,7 stage sensitivity.

Expected data layout under --data-root:
    <dataset>/ExpressionData1000.csv
    <dataset>/PseudoTime.csv
    <dataset>/gold_augmented.csv
"""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
DATASETS=["hESC","hHep","mDC","mESC","mHSC-E","mHSC-GM"]
BINS=[3,4,5,6,7]


def validate_dataset(data_root: Path, dataset: str):
    base=data_root/dataset
    missing=[str(base/x) for x in ["ExpressionData1000.csv","PseudoTime.csv","gold_augmented.csv"] if not (base/x).exists()]
    if missing: raise FileNotFoundError("Missing stage-sensitivity input(s):\n"+"\n".join(missing))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[2])
    ap.add_argument('--data-root',type=Path,required=True)
    ap.add_argument('--output-root',type=Path,default=None)
    ap.add_argument('--datasets',nargs='*',default=None)
    ap.add_argument('--stages',nargs='*',type=int,default=None)
    args=ap.parse_args(); root=args.repo_root.resolve(); out=(args.output_root or root).resolve(); data=args.data_root.resolve()
    ds_list=args.datasets or DATASETS; stages=args.stages or BINS
    invalid=sorted(set(ds_list)-set(DATASETS))
    if invalid: raise ValueError(f"Invalid signed-task dataset(s): {invalid}")
    invalid_t=sorted(set(stages)-set(BINS))
    if invalid_t: raise ValueError(f"Formal sensitivity stages are T=3..7; got {invalid_t}")
    for ds in ds_list: validate_dataset(data,ds)
    logs=out/'stage_sensitivity_logs'; logs.mkdir(parents=True,exist_ok=True)
    script=root/'src'/'train_psdgrn_3c.py'
    for ds in ds_list:
        for t in stages:
            cmd=[sys.executable,'-u',str(script),'--dataset',ds,'--expression_file','ExpressionData1000.csv',
                 '--data_root',str(data),'--output_root',str(out),'--num_classes','3','--K','1','--q','0.1',
                 '--hidden','64','--bins',str(t),'--lr','0.001','--weight_decay','0.05','--dropout','0.5',
                 '--epochs','1000','--patience','20','--checkpoint','25','--train_ratio','0.8','--val_ratio','0.1',
                 '--runs','5','--seed','0','--method','PSD-GRN']
            log=logs/f'{ds}_T{t}.log'; print(f'[RUN] {ds} T={t}')
            with log.open('w',encoding='utf-8') as f: p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
            if p.returncode: raise SystemExit(f'Failed {ds} T={t}; see {log}')
    print(f'Completed {len(ds_list)*len(stages)} configurations.')

if __name__=='__main__': main()

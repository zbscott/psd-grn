#!/usr/bin/env python3
"""Inference from a five-model PSD-GRN ensemble for all ordered pairs or a candidate list.

For --mode all, writes one consensus row per ordered pair (suitable for discovery_rewiring.py).
For --mode candidates, writes per-split probabilities (suitable for external_replication.py).
"""
from __future__ import annotations

import argparse, csv, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from psdgrn_runtime import CLASS_NAMES, load_ensemble, ordered_pair_chunk, preencode  # noqa: E402


def predict_chunk(models, encoded, pairs, device):
    q = torch.tensor(pairs, dtype=torch.long, device=device)
    probs=[]
    with torch.no_grad():
        for m,(hr,hi) in zip(models,encoded):
            probs.append(m.decode(hr,hi,q).exp().cpu().numpy())
    return np.stack(probs, axis=0)  # model x pair x class


def all_pairs(args, models, stage_feat, genes):
    n=len(genes); total=n*(n-1); device=stage_feat.device
    encoded=preencode(models,stage_feat)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["source","target","consensus_state","consensus_votes","mean_p_no_edge","mean_p_activation","mean_p_repression"])
        for start in range(0,total,args.chunk_size):
            stop=min(start+args.chunk_size,total)
            pairs=ordered_pair_chunk(start,stop,n)
            p=predict_chunk(models,encoded,pairs,device)
            pred=np.argmax(p,axis=2)
            mean=p.mean(axis=0)
            for j,(s,t) in enumerate(pairs):
                votes=np.bincount(pred[:,j],minlength=3); c=int(np.argmax(votes))
                w.writerow([genes[s],genes[t],CLASS_NAMES[c],int(votes[c]),*map(float,mean[j])])
            print(f"{stop}/{total}", flush=True)


def candidate_pairs(args, models, stage_feat, genes):
    cand=pd.read_csv(args.candidates)
    cand["source"]=cand["source"].astype(str).str.upper(); cand["target"]=cand["target"].astype(str).str.upper()
    idx={g:i for i,g in enumerate(genes)}
    rows=[]; pairs=[]
    for r in cand[["source","target"]].itertuples(index=False):
        if r.source in idx and r.target in idx:
            rows.append((r.source,r.target)); pairs.append((idx[r.source],idx[r.target]))
    if not pairs:
        raise ValueError("No candidate pairs map to the trajectory gene space.")
    pairs=np.asarray(pairs,dtype=np.int64); device=stage_feat.device; encoded=preencode(models,stage_feat)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    out=[]
    for start in range(0,len(pairs),args.chunk_size):
        stop=min(start+args.chunk_size,len(pairs)); p=predict_chunk(models,encoded,pairs[start:stop],device)
        for m in range(p.shape[0]):
            for j,(source,target) in enumerate(rows[start:stop]):
                out.append({"source":source,"target":target,"split":m,"p_no_edge":p[m,j,0],"p_activation":p[m,j,1],"p_repression":p[m,j,2]})
    pd.DataFrame(out).to_csv(args.out,index=False)
    missing=len(cand)-len(rows)
    print(f"Mapped {len(rows)}/{len(cand)} candidates; missing={missing}")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",required=True)
    ap.add_argument("--expression",required=True,type=Path)
    ap.add_argument("--pseudotime",required=True,type=Path)
    ap.add_argument("--gold",required=True,type=Path)
    ap.add_argument("--output-root",required=True,type=Path)
    ap.add_argument("--mode",choices=["all","candidates"],required=True)
    ap.add_argument("--candidates",type=Path)
    ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--chunk-size",type=int,default=100000)
    ap.add_argument("--bins",type=int,default=5)
    ap.add_argument("--hidden",type=int,default=64)
    ap.add_argument("--K",type=int,default=1)
    ap.add_argument("--q",type=float,default=0.1,help="q/pi fraction; 0.1 means phase q=0.1*pi")
    ap.add_argument("--dropout",type=float,default=0.5)
    ap.add_argument("--device",default=None)
    args=ap.parse_args()
    if args.mode=="candidates" and args.candidates is None:
        ap.error("--candidates is required in candidates mode")
    models,stage_feat,genes,_=load_ensemble(args.expression,args.pseudotime,args.gold,args.output_root,args.dataset,bins=args.bins,hidden=args.hidden,K=args.K,q_fraction=args.q,dropout=args.dropout,device=args.device)
    print(f"Loaded {len(models)} models for {len(genes)} genes on {stage_feat.device}")
    if args.mode=="all": all_pairs(args,models,stage_feat,genes)
    else: candidate_pairs(args,models,stage_feat,genes)

if __name__=="__main__": main()

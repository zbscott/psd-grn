#!/usr/bin/env python3
"""RBPJ/SHARP chromatin-context statistics used for the SPEN-centered analysis.

Input CSV contains one row per locus and columns:
    gene,is_spen_candidate,promoter_peak,cis_peak,wt_signal,ko30_signal,ko36_signal
Boolean columns may be 0/1, True/False, Yes/No.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu


def as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin({"1","true","yes","y"})


def fisher_enrichment(x: pd.DataFrame, col: str) -> dict:
    cand = x["is_spen_candidate"]
    a = int(x.loc[cand, col].sum()); b = int(cand.sum() - a)
    c = int(x.loc[~cand, col].sum()); d = int((~cand).sum() - c)
    odds, p = fisher_exact([[a,b],[c,d]], alternative="greater")
    return {"endpoint":col,"candidate_present":a,"candidate_total":int(cand.sum()),"background_present":c,"background_total":int((~cand).sum()),"odds_ratio":odds,"p_one_sided":p}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loci", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    args = ap.parse_args(); args.outdir.mkdir(parents=True, exist_ok=True)
    x = pd.read_csv(args.loci)
    req = {"gene","is_spen_candidate","promoter_peak","cis_peak","wt_signal","ko30_signal","ko36_signal"}
    if not req.issubset(x.columns):
        raise ValueError(f"Missing columns: {sorted(req.difference(x.columns))}")
    for c in ["is_spen_candidate","promoter_peak","cis_peak"]:
        x[c] = as_bool(x[c])
    for c in ["wt_signal","ko30_signal","ko36_signal"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    fish = pd.DataFrame([fisher_enrichment(x,"promoter_peak"), fisher_enrichment(x,"cis_peak")])
    fish.to_csv(args.outdir / "rbpj_peak_enrichment.csv", index=False)

    x["ko30_wt_ratio"] = x["ko30_signal"] / x["wt_signal"]
    x["ko36_wt_ratio"] = x["ko36_signal"] / x["wt_signal"]
    x["joint_ko_wt_ratio"] = ((x["ko30_signal"] + x["ko36_signal"]) / 2.0) / x["wt_signal"]
    cand = x[x["is_spen_candidate"]]["joint_ko_wt_ratio"].dropna()
    bg = x[~x["is_spen_candidate"]]["joint_ko_wt_ratio"].dropna()
    # Lower KO/WT ratio means a larger reduction after SHARP/SPEN loss.
    u, p = mannwhitneyu(cand, bg, alternative="less")
    stats = pd.DataFrame([{
        "candidate_n":len(cand), "background_n":len(bg),
        "candidate_median_joint_KO_WT":float(cand.median()),
        "background_median_joint_KO_WT":float(bg.median()),
        "wilcoxon_rank_sum_U":float(u), "p_one_sided_candidate_lower":float(p),
    }])
    stats.to_csv(args.outdir / "rbpj_signal_ratio_test.csv", index=False)
    x.to_csv(args.outdir / "rbpj_locus_ratios.csv", index=False)
    print(fish.to_string(index=False))
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()

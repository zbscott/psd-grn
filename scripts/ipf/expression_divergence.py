#!/usr/bin/env python3
"""Stage-wise expression-profile divergence analysis for frozen discovery candidates."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu


def load_expression(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path, index_col=0)
    x.index = x.index.astype(str).str.strip().str.upper()
    x.columns = x.columns.astype(str).str.strip()
    x = x.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return x


def load_pt(path: Path) -> pd.DataFrame:
    pt = pd.read_csv(path)
    if pt.shape[1] < 2:
        raise ValueError("Pseudotime CSV must contain cell and pseudotime columns.")
    lower = {str(c).strip().lower(): c for c in pt.columns}
    cell = lower.get("cell", lower.get("barcode", pt.columns[0]))
    time = lower.get("pseudotime", lower.get("ptime", pt.columns[1]))
    out = pt[[cell,time]].copy(); out.columns=["cell","pseudotime"]
    out["cell"] = out["cell"].astype(str).str.strip()
    out["pseudotime"] = pd.to_numeric(out["pseudotime"], errors="coerce")
    return out.dropna().drop_duplicates("cell").sort_values("pseudotime")


def stage_means(expr: pd.DataFrame, pt: pd.DataFrame, n_stages: int = 5) -> pd.DataFrame:
    cells = [c for c in pt["cell"] if c in expr.columns]
    if not cells:
        raise ValueError("No shared cells between expression matrix and pseudotime table.")
    pt = pt[pt["cell"].isin(cells)].sort_values("pseudotime")
    expr = np.log2(expr[pt["cell"].tolist()] + 1.0)
    bins = np.array_split(np.arange(expr.shape[1]), n_stages)
    arr = np.column_stack([expr.iloc[:, idx].mean(axis=1).to_numpy() for idx in bins])
    return pd.DataFrame(arr, index=expr.index, columns=[f"stage_{i+1}" for i in range(n_stages)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repair-expression", required=True, type=Path)
    ap.add_argument("--repair-pseudotime", required=True, type=Path)
    ap.add_argument("--disease-expression", required=True, type=Path)
    ap.add_argument("--disease-pseudotime", required=True, type=Path)
    ap.add_argument("--candidates", required=True, type=Path)
    ap.add_argument("--external-replication", type=Path, default=None)
    ap.add_argument("--stages", type=int, default=5)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    r = stage_means(load_expression(args.repair_expression), load_pt(args.repair_pseudotime), args.stages)
    d = stage_means(load_expression(args.disease_expression), load_pt(args.disease_pseudotime), args.stages)
    genes = r.index.intersection(d.index)
    r = r.loc[genes]; d = d.loc[genes]

    # Jointly standardize the ten branch-specific stage means for each gene.
    concat = np.concatenate([r.to_numpy(float), d.to_numpy(float)], axis=1)
    mu = concat.mean(axis=1, keepdims=True)
    sd = concat.std(axis=1, keepdims=True)
    sd[sd < 1e-12] = 1.0
    z = (concat - mu) / sd
    gene_div = np.abs(z[:, :args.stages] - z[:, args.stages:]).mean(axis=1)
    gene_score = pd.Series(gene_div, index=genes, name="gene_divergence")

    c = pd.read_csv(args.candidates)
    c["source"] = c["source"].astype(str).str.upper(); c["target"] = c["target"].astype(str).str.upper()
    c["source_divergence"] = c["source"].map(gene_score)
    c["target_divergence"] = c["target"].map(gene_score)
    c["pair_divergence"] = c[["source_divergence","target_divergence"]].mean(axis=1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    c.to_csv(args.out, index=False)

    stats = []
    if "ranking_score" in c.columns:
        valid = c[["pair_divergence","ranking_score"]].dropna()
        rho, p = spearmanr(valid["pair_divergence"], valid["ranking_score"])
        stats.append({"test":"Spearman pair divergence vs ranking score", "statistic":rho, "p":p})

    if args.external_replication is not None:
        ext = pd.read_csv(args.external_replication)
        ext["source"] = ext["source"].astype(str).str.upper(); ext["target"] = ext["target"].astype(str).str.upper()
        x = c[["source","target","pair_divergence"]].merge(
            ext[["source","target","external_evaluable","exact_transition_replication"]], on=["source","target"], how="inner"
        )
        x = x[x["external_evaluable"].astype(bool)].dropna(subset=["pair_divergence"])
        exact = x["exact_transition_replication"].astype(bool)
        u, p = mannwhitneyu(x.loc[exact,"pair_divergence"], x.loc[~exact,"pair_divergence"], alternative="two-sided")
        q75 = float(x["pair_divergence"].quantile(0.75))
        stats.append({
            "test":"Mann-Whitney exact vs non-exact divergence", "statistic":u, "p":p,
            "exact_median":float(x.loc[exact,"pair_divergence"].median()),
            "nonexact_median":float(x.loc[~exact,"pair_divergence"].median()),
            "upper_quartile_threshold":q75,
            "exact_in_upper_quartile":int((x.loc[exact,"pair_divergence"] >= q75).sum()),
            "exact_total":int(exact.sum()),
        })
    if stats:
        pd.DataFrame(stats).to_csv(args.out.with_name(args.out.stem + "_stats.csv"), index=False)
        print(pd.DataFrame(stats).to_string(index=False))


if __name__ == "__main__":
    main()

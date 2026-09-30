#!/usr/bin/env python3
"""Exploratory source-level enrichment of external exact replication."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


def as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin({"1","true","yes","y"})


def bh_adjust(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty_like(q)
    out[order] = q
    return out


def source_tests(df: pd.DataFrame, subset_name: str) -> pd.DataFrame:
    x = df[df["external_evaluable"]].copy()
    exact = x["exact_transition_replication"].astype(bool)
    total_exact = int(exact.sum())
    rows = []
    for source, g in x.groupby("source", sort=True):
        a = int(g["exact_transition_replication"].sum())
        b = int(len(g) - a)
        others = x[x["source"] != source]
        c = int(others["exact_transition_replication"].sum())
        d = int(len(others) - c)
        odds, p = fisher_exact([[a, b], [c, d]], alternative="greater")
        rows.append({
            "subset": subset_name,
            "source": source,
            "candidates_evaluated": len(g),
            "exact_replications": a,
            "exact_replication_rate": a / len(g),
            "odds_ratio": odds,
            "fisher_p_one_sided": p,
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["bh_q"] = bh_adjust(out["fisher_p_one_sided"].to_numpy())
        out = out.sort_values(["bh_q", "fisher_p_one_sided", "source"], kind="mergesort")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replication", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.replication)
    df["source"] = df["source"].astype(str).str.upper()

    all_out = source_tests(df, "all_evaluable_candidates")
    all_out.to_csv(args.outdir / "source_enrichment_all.csv", index=False)

    if "rewiring_class" in df.columns:
        gains = df[df["rewiring_class"] == "Disease-associated gain"].copy()
        gain_out = source_tests(gains, "disease_associated_gains")
        gain_out.to_csv(args.outdir / "source_enrichment_gains.csv", index=False)
    else:
        gain_out = pd.DataFrame()

    if "source_route" in df.columns:
        route = df[as_bool(df["source_route"])].copy()
        route_out = source_tests(route, "source_route_candidates")
        route_out.to_csv(args.outdir / "source_enrichment_source_route.csv", index=False)

    print(f"All-source tests: {len(all_out)}")
    if len(gain_out):
        print(f"Gain-subset tests: {len(gain_out)}")
    sig = all_out[all_out["bh_q"] < 0.05]
    if len(sig):
        print("BH q < 0.05:")
        print(sig.to_string(index=False))


if __name__ == "__main__":
    main()

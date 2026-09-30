#!/usr/bin/env python3
"""Summarize five donor-balanced paired resamplings in GSE135893.

Input state CSV columns:
    source,target,repeat,repair_state,disease_state
Each candidate should have five paired repeats. The frozen discovery table must contain
repair_consensus_state and disease_consensus_state.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--discovery", required=True, type=Path)
    ap.add_argument("--repeat-states", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--external-replication", type=Path, default=None)
    ap.add_argument("--robust-threshold", type=int, default=4)
    ap.add_argument("--expected-repeats", type=int, default=5)
    args = ap.parse_args()

    disc = pd.read_csv(args.discovery)
    states = pd.read_csv(args.repeat_states)
    for df in (disc, states):
        df["source"] = df["source"].astype(str).str.strip().str.upper()
        df["target"] = df["target"].astype(str).str.strip().str.upper()
    req = {"source","target","repeat","repair_state","disease_state"}
    if not req.issubset(states.columns):
        raise ValueError(f"repeat-states missing {sorted(req.difference(states.columns))}")
    reqd = {"source","target","repair_consensus_state","disease_consensus_state"}
    if not reqd.issubset(disc.columns):
        raise ValueError(f"discovery missing {sorted(reqd.difference(disc.columns))}")

    x = states.merge(disc[list(reqd)], on=["source","target"], how="inner", validate="many_to_one")
    x["exact_transition_match"] = (
        (x["repair_state"] == x["repair_consensus_state"]) &
        (x["disease_state"] == x["disease_consensus_state"])
    )
    counts = x.groupby(["source","target"], as_index=False).agg(
        donor_balanced_exact_matches=("exact_transition_match","sum"),
        donor_balanced_repeats=("repeat","nunique"),
    )
    if (counts["donor_balanced_repeats"] != args.expected_repeats).any():
        bad = counts[counts["donor_balanced_repeats"] != args.expected_repeats]
        raise ValueError(f"Candidates with unexpected repeat count:\n{bad.head()}")
    counts["donor_balanced_robust"] = counts["donor_balanced_exact_matches"] >= args.robust_threshold
    out = disc.merge(counts, on=["source","target"], how="left")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Robust >= {args.robust_threshold}/{args.expected_repeats}: {int(out['donor_balanced_robust'].fillna(False).sum())}/{len(out)}")

    if args.external_replication is not None:
        ext = pd.read_csv(args.external_replication)
        ext["source"] = ext["source"].astype(str).str.upper()
        ext["target"] = ext["target"].astype(str).str.upper()
        z = out.merge(
            ext[["source","target","external_evaluable","exact_transition_replication","split_level_exact_matches"]],
            on=["source","target"], how="inner"
        )
        z = z[z["external_evaluable"].astype(bool)].copy()
        robust = z["donor_balanced_robust"].astype(bool)
        a = int(z.loc[robust, "exact_transition_replication"].sum())
        b = int(robust.sum() - a)
        c = int(z.loc[~robust, "exact_transition_replication"].sum())
        d = int((~robust).sum() - c)
        odds, p_f = fisher_exact([[a,b],[c,d]], alternative="two-sided")
        u, p_u = mannwhitneyu(
            z.loc[robust, "split_level_exact_matches"],
            z.loc[~robust, "split_level_exact_matches"],
            alternative="two-sided",
        )
        rho, p_s = spearmanr(z["donor_balanced_exact_matches"], z["split_level_exact_matches"])
        stats = pd.DataFrame([{
            "robust_exact": a, "robust_total": int(robust.sum()),
            "nonrobust_exact": c, "nonrobust_total": int((~robust).sum()),
            "fisher_odds_ratio": odds, "fisher_p_two_sided": p_f,
            "mannwhitney_u": u, "mannwhitney_p_two_sided": p_u,
            "spearman_rho": rho, "spearman_p": p_s,
        }])
        stats.to_csv(args.out.with_name(args.out.stem + "_external_stats.csv"), index=False)
        print(stats.to_string(index=False))


if __name__ == "__main__":
    main()

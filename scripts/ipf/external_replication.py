#!/usr/bin/env python3
"""Evaluate exact external replication of frozen discovery rewiring candidates.

Primary endpoint: for each external trajectory, average three-class probabilities across
five independently trained models, assign the state with the highest mean probability,
and count a candidate as exact only when both external trajectory states equal the two
frozen discovery states. No split-voting threshold is applied to this endpoint.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

CLASS_NAMES = np.array(["No recorded edge", "Activation", "Repression"], dtype=object)
STATE_TO_INT = {x: i for i, x in enumerate(CLASS_NAMES)}


def read_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    req = {"source", "target", "split", "p_no_edge", "p_activation", "p_repression"}
    miss = req.difference(df.columns)
    if miss:
        raise ValueError(f"{path}: missing {sorted(miss)}")
    df = df.copy()
    df["source"] = df["source"].astype(str).str.strip().str.upper()
    df["target"] = df["target"].astype(str).str.strip().str.upper()
    return df


def ensemble(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = read_predictions(path)
    prob_cols = ["p_no_edge", "p_activation", "p_repression"]
    df["split_state_int"] = np.argmax(df[prob_cols].to_numpy(float), axis=1)
    df["split_state"] = CLASS_NAMES[df["split_state_int"].to_numpy(int)]
    g = df.groupby(["source", "target"], as_index=False)[prob_cols].mean()
    p = g[prob_cols].to_numpy(float)
    g["ensemble_state_int"] = np.argmax(p, axis=1)
    g["ensemble_state"] = CLASS_NAMES[g["ensemble_state_int"].to_numpy(int)]
    return g, df


def exact_replication(discovery: pd.DataFrame, repair_path: Path, disease_path: Path) -> pd.DataFrame:
    d = discovery.copy()
    d["source"] = d["source"].astype(str).str.strip().str.upper()
    d["target"] = d["target"].astype(str).str.strip().str.upper()
    if "repair_consensus_state" not in d or "disease_consensus_state" not in d:
        raise ValueError("Discovery table needs repair_consensus_state and disease_consensus_state columns.")

    er, sr = ensemble(repair_path)
    ed, sd = ensemble(disease_path)
    er = er.add_prefix("ext_repair_").rename(columns={"ext_repair_source":"source", "ext_repair_target":"target"})
    ed = ed.add_prefix("ext_disease_").rename(columns={"ext_disease_source":"source", "ext_disease_target":"target"})
    out = d.merge(er, on=["source","target"], how="left").merge(ed, on=["source","target"], how="left")
    out["external_evaluable"] = out["ext_repair_ensemble_state"].notna() & out["ext_disease_ensemble_state"].notna()
    out["exact_transition_replication"] = (
        out["external_evaluable"] &
        (out["ext_repair_ensemble_state"] == out["repair_consensus_state"]) &
        (out["ext_disease_ensemble_state"] == out["disease_consensus_state"])
    )

    # Split-level exact matches: paired split id is used when both tables contain the same split labels.
    keys = ["source","target","split"]
    ss = sr[[*keys,"split_state"]].rename(columns={"split_state":"ext_repair_split_state"})
    tt = sd[[*keys,"split_state"]].rename(columns={"split_state":"ext_disease_split_state"})
    pair = ss.merge(tt, on=keys, how="inner")
    pair = pair.merge(d[["source","target","repair_consensus_state","disease_consensus_state"]], on=["source","target"], how="inner")
    pair["split_exact"] = (
        (pair["ext_repair_split_state"] == pair["repair_consensus_state"]) &
        (pair["ext_disease_split_state"] == pair["disease_consensus_state"])
    )
    counts = pair.groupby(["source","target"], as_index=False)["split_exact"].sum().rename(columns={"split_exact":"split_level_exact_matches"})
    out = out.merge(counts, on=["source","target"], how="left")
    return out


def empirical_null(out: pd.DataFrame, n_perm: int, seed: int, source_preserving: bool = False) -> dict:
    x = out[out["external_evaluable"]].copy().reset_index(drop=True)
    if len(x) == 0:
        raise ValueError("No evaluable candidates.")
    obs = int(x["exact_transition_replication"].sum())
    ext_r = x["ext_repair_ensemble_state"].to_numpy(object)
    ext_d = x["ext_disease_ensemble_state"].to_numpy(object)
    disc_pairs = list(zip(x["repair_consensus_state"], x["disease_consensus_state"]))
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=np.int32)

    if source_preserving:
        groups = [idx.to_numpy() for _, idx in x.groupby("source").groups.items()]
        base_r = np.array([a for a, _ in disc_pairs], dtype=object)
        base_d = np.array([b for _, b in disc_pairs], dtype=object)
        for k in range(n_perm):
            pr = base_r.copy(); pd_ = base_d.copy()
            for idx in groups:
                if len(idx) > 1:
                    perm = rng.permutation(idx)
                    pr[idx] = base_r[perm]
                    pd_[idx] = base_d[perm]
            null[k] = np.sum((ext_r == pr) & (ext_d == pd_))
    else:
        base_r = np.array([a for a, _ in disc_pairs], dtype=object)
        base_d = np.array([b for _, b in disc_pairs], dtype=object)
        for k in range(n_perm):
            perm = rng.permutation(len(x))
            null[k] = np.sum((ext_r == base_r[perm]) & (ext_d == base_d[perm]))

    p = (1 + int(np.sum(null >= obs))) / (n_perm + 1)
    return {
        "observed": obs,
        "null_mean": float(null.mean()),
        "null_q2_5": float(np.quantile(null, 0.025)),
        "null_q97_5": float(np.quantile(null, 0.975)),
        "empirical_p": float(p),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--discovery", type=Path, required=True)
    ap.add_argument("--repair", type=Path, required=True)
    ap.add_argument("--disease", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-perm", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=20260907)
    args = ap.parse_args()

    discovery = pd.read_csv(args.discovery)
    out = exact_replication(discovery, args.repair, args.disease)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    n_eval = int(out['external_evaluable'].sum())
    n_exact = int(out['exact_transition_replication'].sum())
    print(f"Evaluable: {n_eval}/{len(out)}")
    print(f"Exact: {n_exact}/{n_eval}")

    # Exact replication by the six frozen discovery transition types.
    if "transition" in out.columns:
        class_summary = (out[out["external_evaluable"]]
            .groupby("transition", as_index=False)
            .agg(candidates_evaluated=("exact_transition_replication", "size"),
                 exact_replications=("exact_transition_replication", "sum")))
        class_summary["exact_replication_rate"] = class_summary["exact_replications"] / class_summary["candidates_evaluated"]
        class_summary.to_csv(args.out.with_name(args.out.stem + "_by_transition.csv"), index=False)

    # Locked exploratory gain-vs-loss comparison when rewiring-class labels are present.
    if "rewiring_class" in out.columns:
        z = out[out["external_evaluable"]].copy()
        gain = z[z["rewiring_class"] == "Disease-associated gain"]
        loss = z[z["rewiring_class"] == "Disease-associated loss"]
        if len(gain) and len(loss):
            a = int(gain["exact_transition_replication"].sum()); b = int(len(gain)-a)
            c = int(loss["exact_transition_replication"].sum()); d = int(len(loss)-c)
            odds, p_f = fisher_exact([[a,b],[c,d]], alternative="two-sided")
            pd.DataFrame([{
                "gain_exact":a,"gain_total":len(gain),"loss_exact":c,"loss_total":len(loss),
                "fisher_odds_ratio":odds,"fisher_p_two_sided":p_f
            }]).to_csv(args.out.with_name(args.out.stem + "_gain_vs_loss.csv"), index=False)

    if args.n_perm > 0:
        global_null = empirical_null(out, args.n_perm, args.seed, source_preserving=False)
        source_null = empirical_null(out, args.n_perm, args.seed + 1, source_preserving=True)
        pd.DataFrame([
            {"null":"global transition-count preserving permutation", **global_null},
            {"null":"source-preserving permutation", **source_null},
        ]).to_csv(args.out.with_name(args.out.stem + "_null_summary.csv"), index=False)
        print("Global null:", global_null)
        print("Source-preserving null:", source_null)


if __name__ == "__main__":
    main()

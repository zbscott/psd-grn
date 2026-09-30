#!/usr/bin/env python3
"""Discover and prioritize trajectory-specific signed-directed rewiring candidates.

This script implements the trajectory-specific regulatory rewiring definitions used in the manuscript.
It consumes per-split three-class probabilities for the same ordered gene pairs in the
repair-associated and disease-associated trajectories.

Expected prediction CSV columns:
    source,target,split,p_no_edge,p_activation,p_repression
Five split rows per ordered pair are expected by default.

Class coding used throughout the repository:
    0 = No recorded edge
    1 = Activation
    2 = Repression
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

CLASS_NAMES = np.array(["No recorded edge", "Activation", "Repression"], dtype=object)
MARKERS_DEFAULT = [
    "SFTPC", "SFTPA1", "AGER", "CAV1", "PDPN", "KRT17", "KRT5", "KRT8",
    "CLDN4", "SOX9", "CDKN1A", "IL11",
]


def _normalise_prediction_table(df: pd.DataFrame) -> pd.DataFrame:
    required = {"source", "target", "split", "p_no_edge", "p_activation", "p_repression"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    out = df.copy()
    out["source"] = out["source"].astype(str).str.strip().str.upper()
    out["target"] = out["target"].astype(str).str.strip().str.upper()
    out["split"] = pd.to_numeric(out["split"], errors="raise").astype(int)
    prob_cols = ["p_no_edge", "p_activation", "p_repression"]
    out[prob_cols] = out[prob_cols].apply(pd.to_numeric, errors="raise")
    if (out[prob_cols].values < -1e-8).any() or (out[prob_cols].values > 1 + 1e-8).any():
        raise ValueError("Probabilities must lie in [0,1].")
    return out


def summarise_trajectory(df: pd.DataFrame, expected_splits: int = 5) -> pd.DataFrame:
    # Large all-pair inference can be supplied as a pre-aggregated consensus table.
    summary_cols = {
        "source", "target", "consensus_state", "consensus_votes",
        "mean_p_no_edge", "mean_p_activation", "mean_p_repression",
    }
    if summary_cols.issubset(df.columns):
        out = df.copy()
        out["source"] = out["source"].astype(str).str.strip().str.upper()
        out["target"] = out["target"].astype(str).str.strip().str.upper()
        out["consensus_votes"] = pd.to_numeric(out["consensus_votes"], errors="raise").astype(int)
        out["consensus_class"] = out["consensus_state"].map({x: i for i, x in enumerate(CLASS_NAMES)})
        if out["consensus_class"].isna().any():
            raise ValueError("Unknown consensus_state in pre-aggregated prediction table.")
        out["consensus_class"] = out["consensus_class"].astype(int)
        out["stable_4of5"] = out["consensus_votes"] >= 4
        probs = out[["mean_p_no_edge", "mean_p_activation", "mean_p_repression"]].to_numpy(float)
        out["consensus_mean_probability"] = probs[np.arange(len(out)), out["consensus_class"].to_numpy()]
        return out[[
            "source", "target", "consensus_class", "consensus_state", "consensus_votes",
            "stable_4of5", "mean_p_no_edge", "mean_p_activation", "mean_p_repression",
            "consensus_mean_probability",
        ]]

    df = _normalise_prediction_table(df)
    prob_cols = ["p_no_edge", "p_activation", "p_repression"]
    pred = np.argmax(df[prob_cols].to_numpy(), axis=1)
    df = df.assign(pred_class=pred)

    rows = []
    for (source, target), g in df.groupby(["source", "target"], sort=False):
        if expected_splits is not None and g["split"].nunique() != expected_splits:
            raise ValueError(
                f"{source}->{target} has {g['split'].nunique()} unique splits; expected {expected_splits}."
            )
        votes = np.bincount(g["pred_class"].to_numpy(dtype=int), minlength=3)
        consensus = int(np.argmax(votes))
        vote_count = int(votes[consensus])
        mean_probs = g[prob_cols].mean(axis=0).to_numpy(float)
        rows.append({
            "source": source,
            "target": target,
            "consensus_class": consensus,
            "consensus_state": CLASS_NAMES[consensus],
            "consensus_votes": vote_count,
            "stable_4of5": vote_count >= 4,
            "mean_p_no_edge": mean_probs[0],
            "mean_p_activation": mean_probs[1],
            "mean_p_repression": mean_probs[2],
            "consensus_mean_probability": mean_probs[consensus],
        })
    return pd.DataFrame(rows)


def _transition_name(r: int, d: int) -> str:
    if r == d:
        return "Retained"
    mapping = {
        (0, 1): "No recorded edge -> Activation",
        (0, 2): "No recorded edge -> Repression",
        (1, 0): "Activation -> No recorded edge",
        (2, 0): "Repression -> No recorded edge",
        (1, 2): "Activation -> Repression",
        (2, 1): "Repression -> Activation",
    }
    return mapping[(r, d)]


def _rewiring_class(transition: str) -> str:
    if transition.startswith("No recorded edge ->"):
        return "Disease-associated gain"
    if transition.endswith("-> No recorded edge"):
        return "Disease-associated loss"
    if transition in {"Activation -> Repression", "Repression -> Activation"}:
        return "Sign change"
    return "Retained"


def combine_trajectories(repair: pd.DataFrame, disease: pd.DataFrame) -> pd.DataFrame:
    r = repair.add_prefix("repair_").rename(columns={"repair_source": "source", "repair_target": "target"})
    d = disease.add_prefix("disease_").rename(columns={"disease_source": "source", "disease_target": "target"})
    x = r.merge(d, on=["source", "target"], how="inner", validate="one_to_one")

    transitions = []
    classes = []
    deltas = []
    ranks = []
    vmins = []
    cmins = []
    for row in x.itertuples(index=False):
        rc = int(row.repair_consensus_class)
        dc = int(row.disease_consensus_class)
        t = _transition_name(rc, dc)
        transitions.append(t)
        classes.append(_rewiring_class(t))

        e_r = float(row.repair_mean_p_activation + row.repair_mean_p_repression)
        e_d = float(row.disease_mean_p_activation + row.disease_mean_p_repression)
        s_r = float(row.repair_mean_p_activation - row.repair_mean_p_repression)
        s_d = float(row.disease_mean_p_activation - row.disease_mean_p_repression)
        if t.startswith("No recorded edge ->"):
            delta = e_d - e_r
        elif t.endswith("-> No recorded edge"):
            delta = e_r - e_d
        elif t == "Activation -> Repression":
            delta = s_r - s_d
        elif t == "Repression -> Activation":
            delta = s_d - s_r
        else:
            delta = 0.0
        vmin = min(int(row.repair_consensus_votes), int(row.disease_consensus_votes))
        cmin = min(float(row.repair_consensus_mean_probability), float(row.disease_consensus_mean_probability))
        rij = delta * (vmin / 5.0) * cmin
        deltas.append(delta)
        vmins.append(vmin)
        cmins.append(cmin)
        ranks.append(rij)

    x["transition"] = transitions
    x["rewiring_class"] = classes
    x["transition_strength"] = deltas
    x["vmin"] = vmins
    x["cmin"] = cmins
    x["ranking_score"] = ranks
    x["stable_both"] = x["repair_stable_4of5"] & x["disease_stable_4of5"]
    return x


def _load_pairs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"source", "target"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} must contain source,target columns")
    df = df[["source", "target"]].copy()
    df["source"] = df["source"].astype(str).str.strip().str.upper()
    df["target"] = df["target"].astype(str).str.strip().str.upper()
    return df.drop_duplicates()


def _load_gene_list(path: Path) -> set[str]:
    df = pd.read_csv(path, header=None)
    return set(df.iloc[:, 0].astype(str).str.strip().str.upper())


def prioritize(
    rewiring: pd.DataFrame,
    reference_pairs: pd.DataFrame,
    regulator_set: set[str],
    markers: set[str],
    top_k: int = 5000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ref_pairs = set(map(tuple, reference_pairs[["source", "target"]].to_numpy()))
    ref_sources = set(reference_pairs["source"])

    x = rewiring.copy()
    x["recorded_in_reference"] = [
        (s, t) in ref_pairs for s, t in x[["source", "target"]].itertuples(index=False, name=None)
    ]
    stable = x[
        x["stable_both"] &
        (x["transition"] != "Retained") &
        (~x["recorded_in_reference"])
    ].copy()
    stable = stable.sort_values("ranking_score", ascending=False, kind="mergesort").reset_index(drop=True)
    stable["stable_rank"] = np.arange(1, len(stable) + 1)

    top = stable.head(top_k)
    source_route = top[
        top["source"].isin(regulator_set) & ~top["source"].isin(ref_sources)
    ].copy()
    source_route["source_route"] = True

    marker_route = stable[stable["target"].isin(markers)].copy()
    marker_route["marker_route"] = True

    keys = ["source", "target"]
    union_keys = pd.concat([
        source_route[keys].assign(source_route=True, marker_route=False),
        marker_route[keys].assign(source_route=False, marker_route=True),
    ], ignore_index=True)
    if len(union_keys) == 0:
        return stable, stable.iloc[0:0].copy()
    route_flags = union_keys.groupby(keys, as_index=False)[["source_route", "marker_route"]].max()
    prioritized = stable.merge(route_flags, on=keys, how="inner", validate="one_to_one")
    prioritized["selection_route"] = np.select(
        [prioritized["source_route"] & prioritized["marker_route"], prioritized["source_route"]],
        ["source+marker", "source"],
        default="marker",
    )
    return stable, prioritized.sort_values("stable_rank").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repair", required=True, type=Path, help="Repair trajectory per-split probability CSV")
    ap.add_argument("--disease", required=True, type=Path, help="Disease trajectory per-split probability CSV")
    ap.add_argument("--reference", required=True, type=Path, help="Reference recorded relationships CSV with source,target")
    ap.add_argument("--regulators", required=True, type=Path, help="One-column BEELINE regulator list")
    ap.add_argument("--markers", type=str, default=",".join(MARKERS_DEFAULT))
    ap.add_argument("--expected-splits", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=5000)
    ap.add_argument("--outdir", required=True, type=Path)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    repair = summarise_trajectory(pd.read_csv(args.repair), args.expected_splits)
    disease = summarise_trajectory(pd.read_csv(args.disease), args.expected_splits)
    combined = combine_trajectories(repair, disease)
    ref = _load_pairs(args.reference)
    regulators = _load_gene_list(args.regulators)
    markers = set(x.strip().upper() for x in args.markers.split(",") if x.strip())
    stable, prioritized = prioritize(combined, ref, regulators, markers, args.top_k)

    repair.to_csv(args.outdir / "repair_consensus.csv", index=False)
    disease.to_csv(args.outdir / "disease_consensus.csv", index=False)
    combined.to_csv(args.outdir / "all_pair_trajectory_comparison.csv", index=False)
    stable.to_csv(args.outdir / "stable_unrecorded_rewiring.csv", index=False)
    prioritized.to_csv(args.outdir / "prioritized_discovery_candidates.csv", index=False)

    print(f"Stable unrecorded rewiring: {len(stable)}")
    print(f"Prioritized candidates: {len(prioritized)}")
    if len(prioritized):
        print(prioritized["rewiring_class"].value_counts().to_string())
        print(prioritized["selection_route"].value_counts().to_string())


if __name__ == "__main__":
    main()

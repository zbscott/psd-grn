#!/usr/bin/env python3
"""Leakage-safe PSD-GRN graph-classification pilot.

Purpose
-------
Test whether a PSD-GRN-style signed/directed magnetic graph encoder is technically
usable for graph-level state classification when graph samples are constructed
without using class labels.

Important
---------
This script uses the hESC time-course demo already committed in the PSD-GRN repo.
It is a method/debug pilot, not the final biological benchmark.
"""
from __future__ import annotations

import copy
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from get_psdgrn_laplacian import build_psdgrn_laplacian

DATA_DIR = ROOT / "data" / "demo" / "hESC"
EXPR_PATH = DATA_DIR / "ExpressionData1000.csv"
PT_PATH = DATA_DIR / "PseudoTime.csv"
GOLD_PATH = DATA_DIR / "gold_augmented.csv"

DEVICE = torch.device("cpu")
HIDDEN = 64
DROPOUT = 0.5
LR = 1e-3
WEIGHT_DECAY = 5e-2
MAX_EPOCHS = 400
PATIENCE = 30
CHEB_K = 1
Q_FRACTION = 0.1
TARGET_PSEUDOSAMPLES = 100


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_hour(cell: str):
    m = re.search(r"_([0-9]{2})h", str(cell))
    return int(m.group(1)) if m else None


def read_inputs():
    expr = pd.read_csv(EXPR_PATH, index_col=0)
    expr.index = expr.index.astype(str).str.strip().str.upper()
    expr.columns = expr.columns.astype(str).str.strip()
    expr = expr.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    pt = pd.read_csv(PT_PATH, index_col=0)
    pt.index = pt.index.astype(str).str.strip()
    if pt.shape[1] != 1:
        raise ValueError(f"Unexpected pseudotime columns: {pt.columns.tolist()}")
    ptime = pd.to_numeric(pt.iloc[:, 0], errors="coerce").dropna()

    common = [c for c in expr.columns if c in ptime.index and parse_hour(c) is not None]
    if not common:
        raise ValueError("No common labeled cells.")
    expr = expr[common]
    ptime = ptime.loc[common]

    # Keep PSD-GRN preprocessing for expression.
    expr = np.log2(expr + 1.0)

    hours = {c: parse_hour(c) for c in common}
    classes = sorted(set(hours.values()))
    class_to_id = {c: i for i, c in enumerate(classes)}
    cell_label = {c: class_to_id[hours[c]] for c in common}

    gold = pd.read_csv(GOLD_PATH)
    gold["source"] = gold["source"].astype(str).str.strip().str.upper()
    gold["target"] = gold["target"].astype(str).str.strip().str.upper()
    gold["sign"] = pd.to_numeric(gold["sign"], errors="coerce")
    gold = gold.dropna(subset=["source", "target", "sign"])
    gold = gold[gold["sign"].isin([-1, 1])].copy()

    gene_to_idx = {g: i for i, g in enumerate(expr.index.tolist())}
    gold = gold[
        gold["source"].isin(gene_to_idx)
        & gold["target"].isin(gene_to_idx)
    ].copy()

    # Match PSD-GRN gold cleaning: remove ordered pairs with sign conflicts,
    # then remove exact duplicates.
    sign_nunique = gold.groupby(["source", "target"])["sign"].nunique()
    conflicts = set(sign_nunique[sign_nunique > 1].index.tolist())
    if conflicts:
        keep = [
            (s, t) not in conflicts
            for s, t in zip(gold["source"].tolist(), gold["target"].tolist())
        ]
        gold = gold[keep].copy()
    gold = gold.drop_duplicates(["source", "target", "sign"]).copy()

    return expr, ptime, common, cell_label, classes, gold, gene_to_idx


def construct_pseudotime_windows(expr, ptime, cells, target_graphs=TARGET_PSEUDOSAMPLES):
    """Construct NON-OVERLAPPING windows using pseudotime only.

    No class label is accepted by this function by design.
    """
    ordered = sorted(cells, key=lambda c: float(ptime.loc[c]))
    n = len(ordered)
    k = max(4, int(round(n / float(target_graphs))))
    k = min(k, n)

    windows = []
    features = []
    for start in range(0, n, k):
        win = ordered[start : start + k]
        # Do not keep a tiny tail as a standalone graph. Merge it backward.
        if len(win) < max(2, k // 2) and windows:
            prev = windows.pop()
            prev_cells = prev + win
            features.pop()
            windows.append(prev_cells)
            features.append(expr[prev_cells].mean(axis=1).to_numpy(np.float32))
        else:
            windows.append(win)
            features.append(expr[win].mean(axis=1).to_numpy(np.float32))

    # Each original cell must occur exactly once.
    flat = [c for w in windows for c in w]
    assert len(flat) == len(set(flat)) == len(ordered)
    assert set(flat) == set(ordered)
    return np.stack(features).astype(np.float32), windows, k


def assign_posthoc_labels(windows, cell_label, n_classes):
    """Assign graph labels AFTER label-free window construction."""
    y = []
    purity = []
    class_hist = []
    for win in windows:
        labels = np.array([cell_label[c] for c in win], dtype=np.int64)
        counts = np.bincount(labels, minlength=n_classes)
        label = int(np.argmax(counts))
        y.append(label)
        purity.append(float(counts[label] / len(labels)))
        class_hist.append(counts.tolist())
    return np.asarray(y, dtype=np.int64), np.asarray(purity), class_hist


def standardize_by_train(train_x, val_x, test_x):
    mu = train_x.mean(axis=0, keepdims=True)
    sd = train_x.std(axis=0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    return (
        ((train_x - mu) / sd).astype(np.float32),
        ((val_x - mu) / sd).astype(np.float32),
        ((test_x - mu) / sd).astype(np.float32),
    )


def get_edges(gold, gene_to_idx, mode):
    pos, neg = [], []
    for row in gold.itertuples(index=False):
        u = gene_to_idx[row.source]
        v = gene_to_idx[row.target]
        sign = int(row.sign)

        if mode == "signed_directed":
            (pos if sign > 0 else neg).append((u, v))
        elif mode == "directed_unsigned":
            pos.append((u, v))
        elif mode == "undirected_unsigned":
            pos.append((u, v))
            if u != v:
                pos.append((v, u))
        elif mode == "no_graph":
            pass
        else:
            raise ValueError(mode)

    def uniq(arr):
        if not arr:
            return np.empty((0, 2), dtype=np.int64)
        return np.array(sorted(set(arr)), dtype=np.int64).reshape(-1, 2)

    return uniq(pos), uniq(neg)


def build_operator(gold, gene_to_idx, n_genes, mode):
    pos, neg = get_edges(gold, gene_to_idx, mode)
    Lr, Li = build_psdgrn_laplacian(
        pos_edges=pos,
        neg_edges=neg,
        num_nodes=n_genes,
        K=CHEB_K,
        q=np.pi * Q_FRACTION,
        device=DEVICE,
        norm=True,
    )
    return Lr, Li


def spmm_batch(L: torch.Tensor, x: torch.Tensor):
    # x: [B,N,H]. Same graph/operator for all samples.
    b, n, h = x.shape
    z = x.permute(1, 0, 2).reshape(n, b * h)
    out = torch.sparse.mm(L, z)
    return out.reshape(n, b, h).permute(1, 0, 2)


class PSDMagneticBlock(nn.Module):
    """Batched version of the PSD-GRN magnetic block, including its biases."""

    def __init__(self, hidden, Lr, Li, dropout):
        super().__init__()
        if len(Lr) != len(Li):
            raise ValueError("Real/imag operator list lengths differ.")
        self.K = len(Lr)
        for k, x in enumerate(Lr):
            self.register_buffer(f"Lr_{k}", x.coalesce())
        for k, x in enumerate(Li):
            self.register_buffer(f"Li_{k}", x.coalesce())

        self.real_linears = nn.ModuleList([
            nn.Linear(hidden, hidden, bias=False) for _ in range(self.K)
        ])
        self.imag_linears = nn.ModuleList([
            nn.Linear(hidden, hidden, bias=False) for _ in range(self.K)
        ])
        self.bias_real = nn.Parameter(torch.zeros(hidden))
        self.bias_img = nn.Parameter(torch.zeros(hidden))
        self.norm_real = nn.LayerNorm(hidden)
        self.norm_img = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hr, hi):
        out_r = torch.zeros_like(hr)
        out_i = torch.zeros_like(hi)
        for k in range(self.K):
            Lr = getattr(self, f"Lr_{k}")
            Li = getattr(self, f"Li_{k}")
            prop_r = spmm_batch(Lr, hr) - spmm_batch(Li, hi)
            prop_i = spmm_batch(Lr, hi) + spmm_batch(Li, hr)
            out_r = out_r + self.real_linears[k](prop_r)
            out_i = out_i + self.imag_linears[k](prop_i)
        out_r = F.relu(self.norm_real(out_r + self.bias_real))
        out_i = F.relu(self.norm_img(out_i + self.bias_img))
        return self.dropout(out_r), self.dropout(out_i)


class GeneIdentityReadout(nn.Module):
    """Identity-aware attention readout for aligned gene nodes.

    All graph variants use the SAME readout and therefore the same parameter count.
    """
    def __init__(self, n_genes, hidden):
        super().__init__()
        self.gene_embedding = nn.Parameter(torch.empty(n_genes, hidden))
        nn.init.normal_(self.gene_embedding, mean=0.0, std=0.02)
        self.attn = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, hr, hi):
        b, n, h = hr.shape
        g = self.gene_embedding.unsqueeze(0).expand(b, -1, -1)
        s = self.attn(torch.cat([hr, hi, g], dim=-1)).squeeze(-1)
        alpha = torch.softmax(s, dim=1)
        zr = torch.sum(alpha.unsqueeze(-1) * hr, dim=1)
        zi = torch.sum(alpha.unsqueeze(-1) * hi, dim=1)
        zg = torch.sum(alpha.unsqueeze(-1) * g, dim=1)
        return torch.cat([zr, zi, zg], dim=-1)


class MatchedGraphClassifier(nn.Module):
    def __init__(self, n_genes, n_classes, Lr, Li, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.real_proj = nn.Linear(1, hidden)
        self.imag_proj = nn.Linear(1, hidden)
        self.block1 = PSDMagneticBlock(hidden, Lr, Li, dropout)
        self.block2 = PSDMagneticBlock(hidden, Lr, Li, dropout)
        self.res_norm_r = nn.LayerNorm(hidden)
        self.res_norm_i = nn.LayerNorm(hidden)
        self.readout = GeneIdentityReadout(n_genes, hidden)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        hr0 = F.relu(self.real_proj(x))
        hi0 = F.relu(self.imag_proj(x))
        hr1, hi1 = self.block1(hr0, hi0)
        hr2, hi2 = self.block2(hr1, hi1)
        hr2 = self.res_norm_r(hr2 + hr1)
        hi2 = self.res_norm_i(hi2 + hi1)
        z = self.readout(hr2, hi2)
        return self.classifier(z)


class MultiClassFocalLoss(nn.Module):
    def __init__(self, alpha, gamma=0.75):
        super().__init__()
        self.register_buffer("alpha", alpha.float())
        self.gamma = gamma

    def forward(self, logits, target):
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, target.view(-1, 1)).squeeze(1)
        pt = log_pt.exp().clamp(1e-8, 1.0)
        return (self.alpha[target] * ((1.0 - pt) ** self.gamma) * (-log_pt)).mean()


def train_eval(model, train_x, train_y, val_x, val_y, test_x, test_y):
    model = model.to(DEVICE)
    tx = torch.tensor(train_x, dtype=torch.float32, device=DEVICE)
    ty = torch.tensor(train_y, dtype=torch.long, device=DEVICE)
    vx = torch.tensor(val_x, dtype=torch.float32, device=DEVICE)
    vy = torch.tensor(val_y, dtype=torch.long, device=DEVICE)
    sx = torch.tensor(test_x, dtype=torch.float32, device=DEVICE)

    n_classes = int(max(train_y.max(), val_y.max(), test_y.max()) + 1)
    counts = np.bincount(train_y, minlength=n_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    alpha = np.sqrt(counts.sum() / (n_classes * counts))
    alpha = alpha / alpha.mean()
    criterion = MultiClassFocalLoss(torch.tensor(alpha, dtype=torch.float32), gamma=0.75).to(DEVICE)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_state = None
    best_val = float("inf")
    stale = 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        opt.zero_grad()
        loss = criterion(model(tx), ty)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            vloss = float(criterion(model(vx), vy).item())

        if vloss < best_val - 1e-6:
            best_val = vloss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(sx).argmax(dim=1).cpu().numpy()

    return {
        "accuracy": float(accuracy_score(test_y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(test_y, pred)),
        "macro_f1": float(f1_score(test_y, pred, average="macro", zero_division=0)),
        "epochs": int(epoch + 1),
    }


def operator_diagnostics(Lr, Li):
    out = []
    for k, (r, i) in enumerate(zip(Lr, Li)):
        out.append({
            "order": k,
            "real_nnz": int(r._nnz()),
            "imag_nnz": int(i._nnz()),
            "real_abs_sum": float(r.values().abs().sum().item()),
            "imag_abs_sum": float(i.values().abs().sum().item()),
        })
    return out


def main():
    expr, ptime, cells, cell_label, classes, gold, gene_to_idx = read_inputs()
    n_genes = expr.shape[0]
    n_classes = len(classes)

    # Critical separation: label-free sample construction, then label assignment.
    X, windows, k = construct_pseudotime_windows(expr, ptime, cells)
    y, purity, class_hist = assign_posthoc_labels(windows, cell_label, n_classes)

    print("DATA", {"genes": n_genes, "cells": len(cells), "classes": classes})
    print("GOLD", {
        "edges": int(len(gold)),
        "activation": int((gold.sign > 0).sum()),
        "repression": int((gold.sign < 0).sum()),
    })
    print("PSEUDOSAMPLES", {
        "target": TARGET_PSEUDOSAMPLES,
        "k": k,
        "n": len(windows),
        "class_counts": np.bincount(y, minlength=n_classes).tolist(),
        "mean_purity": float(purity.mean()),
        "min_purity": float(purity.min()),
        "mixed_windows": int((purity < 1.0).sum()),
    })

    if np.min(np.bincount(y, minlength=n_classes)) < 5:
        raise ValueError("Too few post-hoc pseudo-samples in at least one class for 5-fold CV.")

    modes = ["no_graph", "undirected_unsigned", "directed_unsigned", "signed_directed"]
    operators = {}
    for mode in modes:
        operators[mode] = build_operator(gold, gene_to_idx, n_genes, mode)
        print("OPERATOR", mode, json.dumps(operator_diagnostics(*operators[mode])))

    # 5-fold outer CV on disjoint pseudo-samples. Validation comes from the training folds only.
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=20261001)
    rows = []

    for fold, (trainval_idx, test_idx) in enumerate(outer.split(X, y), start=1):
        tv_y = y[trainval_idx]
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=31000 + fold)
        inner_train_rel, val_rel = next(inner.split(X[trainval_idx], tv_y))
        train_idx = trainval_idx[inner_train_rel]
        val_idx = trainval_idx[val_rel]

        train_x, val_x, test_x = standardize_by_train(
            X[train_idx], X[val_idx], X[test_idx]
        )
        train_y, val_y, test_y = y[train_idx], y[val_idx], y[test_idx]

        print("FOLD", fold, {
            "train_n": len(train_idx),
            "val_n": len(val_idx),
            "test_n": len(test_idx),
            "train_class": np.bincount(train_y, minlength=n_classes).tolist(),
            "test_class": np.bincount(test_y, minlength=n_classes).tolist(),
        })

        # Critical paired design: SAME initialization seed for every variant in this fold.
        fold_seed = 50000 + fold
        for mode in modes:
            seed_all(fold_seed)  # before model construction
            Lr, Li = operators[mode]
            model = MatchedGraphClassifier(
                n_genes=n_genes,
                n_classes=n_classes,
                Lr=Lr,
                Li=Li,
                hidden=HIDDEN,
                dropout=DROPOUT,
            )
            n_params = sum(p.numel() for p in model.parameters())
            metrics = train_eval(
                model,
                train_x,
                train_y,
                val_x,
                val_y,
                test_x,
                test_y,
            )
            row = {
                "fold": fold,
                "model": mode,
                "seed": fold_seed,
                "params": int(n_params),
                **metrics,
            }
            rows.append(row)
            print("RESULT", json.dumps(row, sort_keys=True))

    df = pd.DataFrame(rows)
    summary = df.groupby("model")[["accuracy", "balanced_accuracy", "macro_f1"]].agg(["mean", "std"])
    print("\nSUMMARY")
    print(summary.to_string())

    outdir = ROOT / "results" / "graph_classification_pilot_v2"
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "per_fold.csv", index=False)
    summary.to_csv(outdir / "summary.csv")

    pd.DataFrame({
        "window_id": np.arange(len(windows)),
        "label": y,
        "purity": purity,
        "cells": ["|".join(w) for w in windows],
        "class_hist": [json.dumps(x) for x in class_hist],
    }).to_csv(outdir / "pseudo_samples.csv", index=False)

    metadata = {
        "dataset": "PSD-GRN hESC time-course demo",
        "purpose": "leakage-safe technical feasibility pilot",
        "sample_construction": "non-overlapping pseudotime-only windows; labels assigned after window construction",
        "target_pseudosamples": TARGET_PSEUDOSAMPLES,
        "k": k,
        "n_pseudosamples": len(windows),
        "mean_label_purity": float(purity.mean()),
        "models": modes,
        "operator": "PSD-GRN build_psdgrn_laplacian, K=1, q=pi*0.1",
        "matched_architecture": True,
        "paired_initialization_within_fold": True,
        "cv": "5-fold stratified pseudo-sample CV; disjoint original cells by construction",
        "hidden": HIDDEN,
        "dropout": DROPOUT,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "classes": classes,
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

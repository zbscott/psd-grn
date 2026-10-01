#!/usr/bin/env python3
import copy
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from get_psdgrn_laplacian import build_psdgrn_laplacian


DATA_DIR = ROOT / "data" / "demo" / "hESC"
EXPR_PATH = DATA_DIR / "ExpressionData1000.csv"
PT_PATH = DATA_DIR / "PseudoTime.csv"
GOLD_PATH = DATA_DIR / "gold_augmented.csv"


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_hour(cell: str):
    m = re.search(r"_([0-9]{2})h", str(cell))
    return int(m.group(1)) if m else None


def load_data():
    expr = pd.read_csv(EXPR_PATH, index_col=0)
    expr.index = expr.index.astype(str).str.strip().str.upper()
    expr.columns = expr.columns.astype(str).str.strip()
    expr = expr.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    expr = np.log2(expr + 1.0)

    pt = pd.read_csv(PT_PATH, index_col=0)
    pt.index = pt.index.astype(str).str.strip()
    pt_col = pt.columns[0]
    ptime = pd.to_numeric(pt[pt_col], errors="coerce").dropna()

    common = [c for c in expr.columns if c in ptime.index and parse_hour(c) is not None]
    expr = expr[common]
    ptime = ptime.loc[common]

    hours = np.array([parse_hour(c) for c in common], dtype=int)
    classes = sorted(np.unique(hours).tolist())
    c2i = {c: i for i, c in enumerate(classes)}
    y = np.array([c2i[h] for h in hours], dtype=int)

    gold = pd.read_csv(GOLD_PATH)
    gold["source"] = gold["source"].astype(str).str.strip().str.upper()
    gold["target"] = gold["target"].astype(str).str.strip().str.upper()
    gold["sign"] = pd.to_numeric(gold["sign"], errors="coerce")
    gold = gold.dropna(subset=["sign"])

    gene_to_idx = {g: i for i, g in enumerate(expr.index.tolist())}
    gold = gold[gold["source"].isin(gene_to_idx) & gold["target"].isin(gene_to_idx)].copy()

    print("DATA_SHAPE", expr.shape)
    print("TIMEPOINTS", classes)
    print("CELL_COUNTS", {int(c): int((hours == c).sum()) for c in classes})
    print("SIGNED_EDGES", int(len(gold)), "POS", int((gold.sign > 0).sum()), "NEG", int((gold.sign < 0).sum()))
    return expr, ptime, np.array(common), y, classes, gold, gene_to_idx


def build_pseudosamples(expr, ptime, cells, labels_by_cell, n_classes, k=8, stride=4):
    xs, ys = [], []
    cell_set = set(cells.tolist() if isinstance(cells, np.ndarray) else cells)
    for cls in range(n_classes):
        cls_cells = [c for c in expr.columns if c in cell_set and labels_by_cell[c] == cls]
        cls_cells.sort(key=lambda c: float(ptime.loc[c]))
        n = len(cls_cells)
        if n < k:
            if n >= 2:
                win = cls_cells
                xs.append(expr[win].mean(axis=1).to_numpy(np.float32))
                ys.append(cls)
            continue
        starts = list(range(0, n - k + 1, stride))
        last = n - k
        if starts[-1] != last:
            starts.append(last)
        for s in starts:
            win = cls_cells[s:s+k]
            xs.append(expr[win].mean(axis=1).to_numpy(np.float32))
            ys.append(cls)
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.int64)


def standardize(train_x, val_x, test_x):
    mu = train_x.mean(axis=0, keepdims=True)
    sd = train_x.std(axis=0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    return (train_x-mu)/sd, (val_x-mu)/sd, (test_x-mu)/sd


def edges_for_mode(gold, gene_to_idx, mode):
    pos, neg = [], []
    for r in gold.itertuples(index=False):
        u = gene_to_idx[r.source]
        v = gene_to_idx[r.target]
        s = float(r.sign)
        if mode == "signed_directed":
            (pos if s > 0 else neg).append((u, v))
        elif mode == "directed_unsigned":
            pos.append((u, v))
        elif mode == "undirected_unsigned":
            pos.append((u, v))
            if u != v:
                pos.append((v, u))
        else:
            raise ValueError(mode)
    def uniq(a):
        return np.array(sorted(set(a)), dtype=np.int64).reshape(-1, 2) if a else np.empty((0, 2), dtype=np.int64)
    return uniq(pos), uniq(neg)


def spmm_batch(L, X):
    # X: [B,N,H], same graph for every sample
    b, n, h = X.shape
    z = X.permute(1, 0, 2).reshape(n, b*h)
    out = torch.sparse.mm(L, z)
    return out.reshape(n, b, h).permute(1, 0, 2)


class MagneticBlock(nn.Module):
    def __init__(self, hidden, Lr, Li, dropout=0.25):
        super().__init__()
        self.Lr, self.Li = Lr, Li
        self.wr = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in Lr])
        self.wi = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in Lr])
        self.nr = nn.LayerNorm(hidden)
        self.ni = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hr, hi):
        orr = torch.zeros_like(hr)
        oii = torch.zeros_like(hi)
        for k in range(len(self.Lr)):
            pr = spmm_batch(self.Lr[k], hr) - spmm_batch(self.Li[k], hi)
            pi = spmm_batch(self.Lr[k], hi) + spmm_batch(self.Li[k], hr)
            orr = orr + self.wr[k](pr)
            oii = oii + self.wi[k](pi)
        orr = self.dropout(F.relu(self.nr(orr)))
        oii = self.dropout(F.relu(self.ni(oii)))
        return orr, oii


class GraphClassifier(nn.Module):
    def __init__(self, n_classes, Lr, Li, hidden=12, dropout=0.25):
        super().__init__()
        self.pr = nn.Linear(1, hidden)
        self.pi = nn.Linear(1, hidden)
        self.b1 = MagneticBlock(hidden, Lr, Li, dropout)
        self.b2 = MagneticBlock(hidden, Lr, Li, dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden*4, hidden*2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden*2, n_classes),
        )

    def forward(self, x):
        # x [B,N]
        x = x.unsqueeze(-1)
        hr = F.relu(self.pr(x))
        hi = F.relu(self.pi(x))
        r1, i1 = self.b1(hr, hi)
        r2, i2 = self.b2(r1 + hr, i1 + hi)
        z = torch.cat([
            r2.mean(dim=1), i2.mean(dim=1),
            r2.max(dim=1).values, i2.max(dim=1).values
        ], dim=1)
        return self.head(z)


class ExpressionMLP(nn.Module):
    def __init__(self, n_genes, n_classes, hidden=128, dropout=0.35):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_genes, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes)
        )
    def forward(self, x):
        return self.net(x)


def fit_eval(model, train_x, train_y, val_x, val_y, test_x, test_y, seed, lr=3e-3, max_epochs=60, patience=10):
    seed_all(seed)
    device = torch.device("cpu")
    model = model.to(device)
    tx = torch.tensor(train_x, dtype=torch.float32, device=device)
    ty = torch.tensor(train_y, dtype=torch.long, device=device)
    vx = torch.tensor(val_x, dtype=torch.float32, device=device)
    vy = torch.tensor(val_y, dtype=torch.long, device=device)
    sx = torch.tensor(test_x, dtype=torch.float32, device=device)

    counts = np.bincount(train_y, minlength=int(train_y.max())+1).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_loss = float("inf")
    best_state = None
    stale = 0
    for epoch in range(max_epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(tx), ty)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            vloss = float(loss_fn(model(vx), vy).item())
        if vloss < best_loss - 1e-5:
            best_loss = vloss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
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
        "epochs": int(epoch+1),
    }


def main():
    expr, ptime, cells, y_cell, classes, gold, gene_to_idx = load_data()
    n_classes = len(classes)
    n_genes = expr.shape[0]
    labels_by_cell = {c: int(y) for c, y in zip(cells, y_cell)}

    # Provisional pilot aggregation only; not a locked paper hyperparameter.
    k = 8
    stride = 4

    modes = ["expression_mlp", "signed_directed"]
    all_rows = []

    outer = StratifiedKFold(n_splits=3, shuffle=True, random_state=20261001)
    for fold, (trv_idx, te_idx) in enumerate(outer.split(cells, y_cell), start=1):
        trv_cells, te_cells = cells[trv_idx], cells[te_idx]
        trv_y = y_cell[trv_idx]
        inner = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=9000+fold)
        tr_rel, va_rel = next(inner.split(trv_cells, trv_y))
        tr_cells, va_cells = trv_cells[tr_rel], trv_cells[va_rel]

        train_x, train_y = build_pseudosamples(expr, ptime, tr_cells, labels_by_cell, n_classes, k, stride)
        val_x, val_y = build_pseudosamples(expr, ptime, va_cells, labels_by_cell, n_classes, k, stride)
        test_x, test_y = build_pseudosamples(expr, ptime, te_cells, labels_by_cell, n_classes, k, stride)
        train_x, val_x, test_x = standardize(train_x, val_x, test_x)

        print("FOLD_SAMPLES", fold, len(train_y), len(val_y), len(test_y),
              "train_class", np.bincount(train_y, minlength=n_classes).tolist(),
              "test_class", np.bincount(test_y, minlength=n_classes).tolist())

        for mi, mode in enumerate(modes):
            run_seed = 10000 + fold*100 + mi
            if mode == "expression_mlp":
                model = ExpressionMLP(n_genes, n_classes)
                metrics = fit_eval(model, train_x, train_y, val_x, val_y, test_x, test_y, run_seed, lr=2e-3)
            else:
                pos, neg = edges_for_mode(gold, gene_to_idx, mode)
                Lr, Li = build_psdgrn_laplacian(
                    pos_edges=pos,
                    neg_edges=neg,
                    num_nodes=n_genes,
                    K=1,
                    q=np.pi*0.1,
                    device=torch.device("cpu"),
                    norm=True,
                )
                model = GraphClassifier(n_classes, Lr, Li)
                metrics = fit_eval(model, train_x, train_y, val_x, val_y, test_x, test_y, run_seed, lr=3e-3)
            row = {"fold": fold, "model": mode, **metrics}
            all_rows.append(row)
            print("RESULT", json.dumps(row, sort_keys=True))

    df = pd.DataFrame(all_rows)
    print("\nSUMMARY")
    summary = df.groupby("model")[["accuracy","balanced_accuracy","macro_f1"]].agg(["mean","std"])
    print(summary.to_string())

    outdir = ROOT / "results" / "graph_classification_quick"
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "per_fold.csv", index=False)
    summary.to_csv(outdir / "summary.csv")
    metadata = {
        "dataset": "hESC time-course demo from PSD-GRN repository",
        "task": "6-class experimental time-point classification (00/12/24/36/72/96h)",
        "warning": "This is a computational feasibility pilot, not the final independent cell-state benchmark.",
        "aggregation": {"k": k, "stride": stride, "split_before_aggregation": True},
        "graph": "fixed PSD-GRN gold_augmented signed-directed prior",
        "outer_cv": "3-fold stratified at original-cell level",
        "classes": classes,
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

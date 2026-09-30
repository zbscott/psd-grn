"""Reusable PSD-GRN model/runtime extracted from the archived training implementation.

The mathematical operations in the stage feature constructor, signed-directed magnetic
Laplacian block, stage-residual integration, and ordered-pair decoder follow the archived
`train_psdgrn_3c.py`. This module exists so inference/analysis scripts can load trained
checkpoints without executing the training script at import time.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

try:
    from .get_psdgrn_laplacian import build_psdgrn_laplacian
except ImportError:  # direct script use with src on PYTHONPATH
    from get_psdgrn_laplacian import build_psdgrn_laplacian


CLASS_NAMES = np.array(["No recorded edge", "Activation", "Repression"], dtype=object)


def read_gold_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df_try = pd.read_csv(path)
    cols_lower = [str(c).strip().lower() for c in df_try.columns]
    col_map = {str(c).strip().lower(): c for c in df_try.columns}
    if {"source", "target", "sign"}.issubset(cols_lower):
        df = df_try[[col_map["source"], col_map["target"], col_map["sign"]]].copy()
        df.columns = ["source", "target", "sign"]
        return df
    if {"target", "sign"}.issubset(cols_lower):
        source_cols = [c for c in df_try.columns if str(c).strip().lower() not in {"target", "sign"}]
        if source_cols:
            df = df_try[[source_cols[0], col_map["target"], col_map["sign"]]].copy()
            df.columns = ["source", "target", "sign"]
            return df
    return pd.read_csv(path, header=None, names=["source", "target", "sign"])


def load_expression_gene_names(expression_path: str | Path, gold_path: str | Path) -> Tuple[pd.DataFrame, List[str]]:
    expr = pd.read_csv(expression_path, header=0, index_col=0)
    expr.index = expr.index.astype(str).str.strip()
    expr.columns = expr.columns.astype(str).str.strip()
    gold = read_gold_table(gold_path)
    gold_genes = set(gold["source"].astype(str).str.strip().str.upper()).union(
        set(gold["target"].astype(str).str.strip().str.upper())
    )
    idx_upper = pd.Index(expr.index).astype(str).str.strip().str.upper()
    col_upper = pd.Index(expr.columns).astype(str).str.strip().str.upper()
    idx_overlap = len(gold_genes.intersection(set(idx_upper)))
    col_overlap = len(gold_genes.intersection(set(col_upper)))
    if idx_overlap == 0 and col_overlap == 0:
        raise ValueError("Gold-standard gene names do not overlap expression rows or columns.")
    if col_overlap > idx_overlap:
        expr = expr.T
    expr.index = expr.index.astype(str).str.strip().str.upper()
    expr.columns = expr.columns.astype(str).str.strip()
    return expr, expr.index.tolist()


def build_one_graph_features(expression_path: str | Path, pseudotime_path: str | Path, num_nodes: int, n_bins: int = 5) -> torch.Tensor:
    # Kept aligned with train_psdgrn_3c.py.
    expr = pd.read_csv(expression_path, header=0, index_col=0)
    expr = expr.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    expr.index = expr.index.astype(str).str.strip().str.upper()
    expr.columns = expr.columns.astype(str).str.strip()
    if expr.shape[0] != num_nodes and expr.shape[1] == num_nodes:
        expr = expr.T
        expr.index = expr.index.astype(str).str.strip().str.upper()
        expr.columns = expr.columns.astype(str).str.strip()
    if expr.shape[0] != num_nodes:
        raise ValueError(f"Expression genes ({expr.shape[0]}) != num_nodes ({num_nodes}).")

    pt = pd.read_csv(pseudotime_path)
    if pt.shape[1] == 1:
        pt = pt.reset_index(); pt.columns = ["cell", "pseudotime"]
    else:
        lower_map = {str(c).strip().lower(): c for c in pt.columns}
        cell_col = next((lower_map[x] for x in ["cell", "cells", "barcode", "barcodes"] if x in lower_map), pt.columns[0])
        time_col = next((lower_map[x] for x in ["pseudotime", "ptime", "time"] if x in lower_map), pt.columns[1])
        pt = pt[[cell_col, time_col]].copy(); pt.columns = ["cell", "pseudotime"]
    pt["cell"] = pt["cell"].astype(str).str.strip()
    pt["pseudotime"] = pd.to_numeric(pt["pseudotime"], errors="coerce")
    pt = pt.dropna().drop_duplicates("cell").sort_values("pseudotime")
    common_cells = [c for c in pt["cell"].tolist() if c in expr.columns]
    if not common_cells:
        raise ValueError("No common cells between expression and pseudotime inputs.")
    pt = pt[pt["cell"].isin(common_cells)].sort_values("pseudotime")
    expr = expr[pt["cell"].tolist()]
    expr = np.log2(expr + 1.0)
    t = pt["pseudotime"].to_numpy(np.float32)
    t = (t - t.min()) / (t.max() - t.min() + 1e-8)
    n_bins = min(n_bins, len(t))
    bins = np.array_split(np.arange(len(t)), n_bins)
    stage = []
    for idx in bins:
        xb = expr.iloc[:, idx].to_numpy(np.float32)
        tb = t[idx].astype(np.float32)
        mean_b = xb.mean(axis=1); std_b = xb.std(axis=1); max_b = xb.max(axis=1); min_b = xb.min(axis=1)
        if len(tb) >= 2 and np.var(tb) > 1e-12:
            tc = tb - tb.mean(); xc = xb - xb.mean(axis=1, keepdims=True)
            slope = (xc @ tc) / (np.sum(tc ** 2) + 1e-8)
        else:
            slope = np.zeros(xb.shape[0], dtype=np.float32)
        stage.append(np.stack([mean_b, std_b, max_b, min_b, slope], axis=1))
    arr = np.stack(stage, axis=1).astype(np.float32)
    flat = arr.reshape(arr.shape[0], -1)
    flat = StandardScaler().fit_transform(flat)
    arr = flat.reshape(arr.shape[0], n_bins, 5).astype(np.float32)
    return torch.tensor(arr, dtype=torch.float32)


def laplacian_matmul(L: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    return torch.sparse.mm(L, X) if L.is_sparse else torch.matmul(L, X)


class PSDGRNStageBlock(nn.Module):
    def __init__(self, hidden_dim, L_norm_real, L_norm_imag, dropout=0.0):
        super().__init__()
        self.L_norm_real = L_norm_real; self.L_norm_imag = L_norm_imag; self.K = len(L_norm_real)
        self.real_linears = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(self.K)])
        self.imag_linears = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(self.K)])
        self.bias_real = nn.Parameter(torch.zeros(hidden_dim)); self.bias_img = nn.Parameter(torch.zeros(hidden_dim))
        self.norm_real = nn.LayerNorm(hidden_dim); self.norm_img = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_real, h_img):
        out_real = 0.0; out_img = 0.0
        for k in range(self.K):
            lr = self.L_norm_real[k]; li = self.L_norm_imag[k]
            prop_real = laplacian_matmul(lr, h_real) - laplacian_matmul(li, h_img)
            prop_img = laplacian_matmul(lr, h_img) + laplacian_matmul(li, h_real)
            out_real = out_real + self.real_linears[k](prop_real)
            out_img = out_img + self.imag_linears[k](prop_img)
        out_real = self.dropout(F.relu(self.norm_real(out_real + self.bias_real)))
        out_img = self.dropout(F.relu(self.norm_img(out_img + self.bias_img)))
        return out_real, out_img


class PSDGRNLinkPredictor(nn.Module):
    def __init__(self, num_features, stage_feature_dim, hidden, L_norm_real, L_norm_imag, label_dim, num_stages=5, dropout=0.0):
        super().__init__()
        self.num_stages = num_stages; self.hidden = hidden
        self.stage_real_proj = nn.ModuleList([nn.Linear(num_features + stage_feature_dim, hidden) for _ in range(num_stages)])
        self.stage_img_proj = nn.ModuleList([nn.Linear(num_features + stage_feature_dim, hidden) for _ in range(num_stages)])
        self.stage_blocks = nn.ModuleList([
            PSDGRNStageBlock(hidden, L_norm_real, L_norm_imag, dropout) for _ in range(num_stages)
        ])
        self.res_norm_real = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_stages)])
        self.res_norm_img = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_stages)])
        edge_dim = hidden * 8
        self.link_predictor = nn.Sequential(nn.Linear(edge_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, label_dim))

    def encode(self, X_real, X_img, one_graph_features):
        device = X_real.device; one_graph_features = one_graph_features.to(device)
        h_real_prev = None; h_img_prev = None
        for i in range(self.num_stages):
            feat = one_graph_features[:, i, :]
            hr = F.relu(self.stage_real_proj[i](torch.cat([X_real, feat], dim=1)))
            hi = F.relu(self.stage_img_proj[i](torch.cat([X_img, feat], dim=1)))
            hr, hi = self.stage_blocks[i](hr, hi)
            if h_real_prev is not None:
                hr = self.res_norm_real[i](hr + h_real_prev)
                hi = self.res_norm_img[i](hi + h_img_prev)
            h_real_prev, h_img_prev = hr, hi
        return h_real_prev, h_img_prev

    def decode(self, h_real, h_img, query_edges):
        if query_edges.dim() != 2:
            raise ValueError(f"query_edges shape {query_edges.shape}")
        if query_edges.size(0) == 2 and query_edges.size(1) != 2:
            query_edges = query_edges.t().contiguous()
        query_edges = query_edges.long().to(h_real.device)
        src = query_edges[:, 0]; dst = query_edges[:, 1]
        zi = torch.cat([h_real[src], h_img[src]], dim=1)
        zj = torch.cat([h_real[dst], h_img[dst]], dim=1)
        edge_repr = torch.cat([zi, zj, zi * zj, torch.abs(zi - zj)], dim=1)
        return F.log_softmax(self.link_predictor(edge_repr), dim=1)

    def forward(self, X_real, X_img, query_edges, one_graph_features):
        hr, hi = self.encode(X_real, X_img, one_graph_features)
        return self.decode(hr, hi, query_edges)


def extract_pos_neg_edges(edge_index: torch.Tensor, edge_weight: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    ei = edge_index.detach().cpu().long().t().numpy()
    ew = edge_weight.detach().cpu().numpy()
    return ei[ew > 0], ei[ew < 0]


def torch_load_any(path: str | Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def find_split_cache(output_root: str | Path, dataset: str, num_nodes: int) -> Path:
    root = Path(output_root) / "data" / dataset
    hits = sorted(root.glob(f"*DirectSign3C_nodes{num_nodes}.pt"), key=lambda p: p.stat().st_mtime)
    if not hits:
        raise FileNotFoundError(f"No signed 3-class split cache under {root} for {num_nodes} nodes")
    return hits[-1]


def checkpoint_path(output_root: str | Path, dataset: str, split: int, method: str = "PSD-GRN") -> Path:
    return Path(output_root) / "logs" / method / dataset / f"model_err{split}.t7"


def load_ensemble(
    expression_path: str | Path,
    pseudotime_path: str | Path,
    gold_path: str | Path,
    output_root: str | Path,
    dataset: str,
    *,
    bins: int = 5,
    hidden: int = 64,
    K: int = 1,
    q_fraction: float = 0.1,
    dropout: float = 0.5,
    device: str | torch.device | None = None,
) -> Tuple[List[PSDGRNLinkPredictor], torch.Tensor, List[str], dict]:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    expr, genes = load_expression_gene_names(expression_path, gold_path)
    num_nodes = len(genes)
    stage_feat = build_one_graph_features(expression_path, pseudotime_path, num_nodes, bins).to(device)
    split_cache = find_split_cache(output_root, dataset, num_nodes)
    link_data = torch_load_any(split_cache, map_location="cpu")
    models = []
    for split in sorted(link_data.keys()):
        edge_index = link_data[split]["graph"].detach().cpu()
        edge_weight = link_data[split]["weights"].detach().cpu().float()
        pos, neg = extract_pos_neg_edges(edge_index, edge_weight)
        lr, li = build_psdgrn_laplacian(pos, neg, num_nodes, K, np.pi * q_fraction, device, norm=True)
        model = PSDGRNLinkPredictor(0, stage_feat.size(2), hidden, lr, li, 3, bins, dropout).to(device)
        cp = checkpoint_path(output_root, dataset, int(split))
        if not cp.exists():
            raise FileNotFoundError(cp)
        state = torch_load_any(cp, map_location=device)
        model.load_state_dict(state); model.eval(); models.append(model)
    return models, stage_feat, genes, link_data


def ordered_pair_chunk(start: int, stop: int, n: int) -> np.ndarray:
    ids = np.arange(start, stop, dtype=np.int64)
    src = ids // (n - 1)
    rem = ids % (n - 1)
    dst = rem + (rem >= src)
    return np.column_stack([src, dst]).astype(np.int64)


def preencode(models: Sequence[PSDGRNLinkPredictor], stage_feat: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    n = stage_feat.size(0); device = stage_feat.device
    xr = torch.empty((n, 0), dtype=torch.float32, device=device); xi = xr.clone()
    encoded = []
    with torch.no_grad():
        for m in models:
            encoded.append(m.encode(xr, xi, stage_feat))
    return encoded

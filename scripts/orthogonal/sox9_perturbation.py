#!/usr/bin/env python3
"""Summarize the author-provided GSE284531 SOX9-knockdown differential-expression table."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

ABERRANT_BASALOID_20 = [
    "TP63", "KRT17", "LAMB3", "LAMC2",
    "VIM", "CDH2", "FN1", "COL1A1", "TNC", "HMGA2",
    "CDKN1A", "CDKN2A", "CCND1", "CCND2", "MDM2", "GDF15",
    "MMP7", "ITGAV", "ITGB6", "EPHB2",
]


def read_deg(path: Path, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet)
    # tolerate common spelling/case variants
    rename = {}
    for c in df.columns:
        lc = str(c).strip().lower()
        if lc == "gene": rename[c] = "Gene"
        elif lc == "logfc": rename[c] = "logFC"
        elif lc in {"fdr", "adj.p.val", "padj"}: rename[c] = "FDR"
    df = df.rename(columns=rename)
    req = {"Gene","logFC","FDR"}
    if not req.issubset(df.columns):
        raise ValueError(f"Sheet {sheet!r} lacks required columns {sorted(req)}. Found {list(df.columns)}")
    df = df.copy()
    df["Gene"] = df["Gene"].astype(str).str.strip().str.upper()
    df["logFC"] = pd.to_numeric(df["logFC"], errors="coerce")
    df["FDR"] = pd.to_numeric(df["FDR"], errors="coerce")
    return df.dropna(subset=["Gene","logFC","FDR"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workbook", required=True, type=Path)
    ap.add_argument("--sheet", default="0.05 FDR DEG")
    ap.add_argument("--outdir", required=True, type=Path)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = read_deg(args.workbook, args.sheet)
    sig = df[df["FDR"] < 0.05].copy()
    down = sig[sig["logFC"] < 0]
    up = sig[sig["logFC"] > 0]
    sox9 = sig[sig["Gene"] == "SOX9"]

    summary = {
        "significant_genes_FDR_lt_0.05": len(sig),
        "decreased": len(down),
        "increased": len(up),
        "SOX9_present": bool(len(sox9)),
        "SOX9_logFC": float(sox9.iloc[0]["logFC"]) if len(sox9) else None,
        "SOX9_FDR": float(sox9.iloc[0]["FDR"]) if len(sox9) else None,
    }
    pd.DataFrame([summary]).to_csv(args.outdir / "sox9_deg_summary.csv", index=False)

    audit = pd.DataFrame({"Gene": ABERRANT_BASALOID_20})
    audit = audit.merge(sig[["Gene","logFC","FDR"]], on="Gene", how="left")
    audit["evaluable_in_filtered_table"] = audit["logFC"].notna()
    audit["direction"] = audit["logFC"].map(lambda x: "Decreased" if pd.notna(x) and x < 0 else ("Increased" if pd.notna(x) and x > 0 else "Unevaluable"))
    audit.to_csv(args.outdir / "aberrant_basaloid_20gene_audit.csv", index=False)

    print(pd.DataFrame([summary]).to_string(index=False))
    print("\n20-gene set matched genes:")
    print(audit[audit["evaluable_in_filtered_table"]].to_string(index=False))


if __name__ == "__main__":
    main()

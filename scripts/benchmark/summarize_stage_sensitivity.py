import argparse
from pathlib import Path
import numpy as np
import pandas as pd

DATASETS = ["hESC", "hHep", "mDC", "mESC", "mHSC-E", "mHSC-GM"]
BINS = [3, 4, 5, 6, 7]
MACRO_F1_COL = 1
AUROC_COL = 4


def locate_result(root: Path, dataset: str, bins: int) -> Path:
    base = root / "link_results" / "sign" / dataset
    pattern = f"**/K1100q10hidden64_PSDGRN_3C_Bins{bins}_MeanStdMaxMinSlope_NoTF_NoDegree_ArticleEdgeConcat.npy"
    hits = sorted(base.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not hits:
        raise FileNotFoundError(f"No result found for {dataset}, T={bins} under {base}")
    return hits[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="PSD_GRN repository root")
    args = ap.parse_args()
    root = Path(args.root).resolve()

    rows = []
    for dataset in DATASETS:
        for bins in BINS:
            path = locate_result(root, dataset, bins)
            arr = np.load(path)
            if arr.ndim != 2 or arr.shape[0] != 5 or arr.shape[1] < 5:
                raise ValueError(f"Unexpected result shape {arr.shape} in {path}; expected 5 x >=5")
            auroc = arr[:, AUROC_COL].astype(float)
            macro = arr[:, MACRO_F1_COL].astype(float)
            rows.append({
                "Dataset": dataset,
                "T": bins,
                "AUROC_mean": np.nanmean(auroc) * 100.0,
                "AUROC_SD_5splits": np.nanstd(auroc, ddof=0) * 100.0,
                "MacroF1_mean": np.nanmean(macro) * 100.0,
                "MacroF1_SD_5splits": np.nanstd(macro, ddof=0) * 100.0,
                "Result_file": str(path),
            })

    dataset_df = pd.DataFrame(rows)
    dataset_csv = root / "stage_sensitivity_dataset_level.csv"
    dataset_df.to_csv(dataset_csv, index=False, encoding="utf-8-sig")

    summary_rows = []
    for bins in BINS:
        x = dataset_df[dataset_df["T"] == bins]
        summary_rows.append({
            "T": bins,
            "AUROC_cross_dataset_mean": x["AUROC_mean"].mean(),
            "AUROC_cross_dataset_SD": x["AUROC_mean"].std(ddof=0),
            "MacroF1_cross_dataset_mean": x["MacroF1_mean"].mean(),
            "MacroF1_cross_dataset_SD": x["MacroF1_mean"].std(ddof=0),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = root / "stage_sensitivity_cross_dataset_summary.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    t5 = summary_df[summary_df["T"] == 5].iloc[0]
    print("\nCross-dataset summary (six dataset-level means; SD ddof=0):")
    print(summary_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nT=5 check against the benchmark cross-dataset means:")
    print(f"AUROC: new T=5 = {t5['AUROC_cross_dataset_mean']:.4f}, locked = 94.8400, difference = {t5['AUROC_cross_dataset_mean']-94.84:+.4f}")
    print(f"Macro-F1: new T=5 = {t5['MacroF1_cross_dataset_mean']:.4f}, locked = 79.5100, difference = {t5['MacroF1_cross_dataset_mean']-79.51:+.4f}")
    print(f"\nSaved: {dataset_csv}")
    print(f"Saved: {summary_csv}")


if __name__ == "__main__":
    main()

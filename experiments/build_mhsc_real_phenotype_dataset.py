#!/usr/bin/env python3
"""Build GSE81682 mHSC trajectory datasets with real FACS phenotype labels.

Sources
-------
1) Expression + pseudotime trajectories: BEELINE copies in CellProphet.
2) Real phenotype annotations: original Nestorowa/Cambridge cell_types.txt,
   derived from FACS/index-sorting gates.

Important protocol rule
-----------------------
Labels are stored for supervision/audit only. Pseudo-samples must be created later
from pseudotime WITHOUT consulting these labels; labels are assigned to windows
only after window construction.
"""
from pathlib import Path
import urllib.request
import pandas as pd
import numpy as np
import hashlib, json, shutil

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/"tmp_mhsc_dataset_build"
OUT=ROOT/"results"/"GSE81682_mHSC_real_phenotype_dataset"
WORK.mkdir(parents=True,exist_ok=True)
if OUT.exists(): shutil.rmtree(OUT)
OUT.mkdir(parents=True)

CELLTYPE_URL="https://blood.stemcells.cam.ac.uk/data/cell_types.txt"
BASE="https://media.githubusercontent.com/media/prsigma/CellProphet/main/BEELINE-data"
TRAJ={
    "mHSC-E":{
        "classes":["LTHSC","MPP","CMP","MEP"],
        "class_order":{"LTHSC":0,"MPP":1,"CMP":2,"MEP":3},
        "description":"erythroid lineage benchmark"
    },
    "mHSC-GM":{
        "classes":["LTHSC","MPP","CMP","GMP"],
        "class_order":{"LTHSC":0,"MPP":1,"CMP":2,"GMP":3},
        "description":"granulocyte/monocyte lineage benchmark"
    },
    "mHSC-L":{
        "classes":["LTHSC","MPP","LMPP"],
        "class_order":{"LTHSC":0,"MPP":1,"LMPP":2},
        "description":"lymphoid lineage benchmark"
    },
}

def download(url,p):
    if not p.exists():
        print("DOWNLOAD",url)
        urllib.request.urlretrieve(url,p)
    print("FILE",p,p.stat().st_size)

def norm_for_labels(c):
    c=str(c)
    if c.startswith("LT_HSC_"):
        return "LT-HSC_"+c[len("LT_HSC_"):]
    return c

def read_bool_table(path):
    df=pd.read_csv(path,sep=r"\s+",engine="python",index_col=0)
    def conv(v):
        s=str(v).strip().upper()
        if s=="TRUE": return 1
        if s=="FALSE": return 0
        try: return int(float(s))
        except: return 0
    return df.map(conv).astype(np.int8)

def macro_from_row(r):
    macro={
        "LTHSC":max(int(r["LTHSC"]),int(r["ESLAM"]),int(r["HSC1"])),
        "LMPP":int(r["LMPP"]),
        "MPP":max(int(r["MPP"]),int(r["MPP1"]),int(r["MPP2"]),int(r["MPP3"]),int(r["STHSC"])),
        "CMP":int(r["CMP"]),
        "MEP":int(r["MEP"]),
        "GMP":int(r["GMP"]),
    }
    pos=[k for k,v in macro.items() if v==1]
    if len(pos)==1: return pos[0],1
    if len(pos)==0: return "",0
    return "AMBIGUOUS",len(pos)

def orient_expression(expr, pt_cells):
    ptset=set(pt_cells)
    col_overlap=sum(c in ptset for c in expr.columns.astype(str))
    idx_overlap=sum(c in ptset for c in expr.index.astype(str))
    if idx_overlap>col_overlap:
        expr=expr.T
    expr.index=expr.index.astype(str)
    expr.columns=expr.columns.astype(str)
    return expr

download(CELLTYPE_URL,WORK/"cell_types.txt")
ct=read_bool_table(WORK/"cell_types.txt")
required=["LTHSC","LMPP","MPP","CMP","MEP","GMP","MPP1","MPP2","MPP3","STHSC","ESLAM","HSC1"]
missing=[x for x in required if x not in ct.columns]
if missing: raise ValueError(f"Missing phenotype columns: {missing}")

root_summary=[]
source_manifest={
    "GSE":"GSE81682",
    "phenotype_source":CELLTYPE_URL,
    "expression_pseudotime_source_repository":"prsigma/CellProphet (BEELINE-data)",
    "cell_type_columns":required,
    "macro_definition":{
        "LTHSC":["LTHSC","ESLAM","HSC1"],
        "LMPP":["LMPP"],
        "MPP":["MPP","MPP1","MPP2","MPP3","STHSC"],
        "CMP":["CMP"],
        "MEP":["MEP"],
        "GMP":["GMP"],
    }
}

for ds,cfg in TRAJ.items():
    dswork=WORK/ds; dswork.mkdir(exist_ok=True)
    expr_p=dswork/"ExpressionData.csv"
    pt_p=dswork/"PseudoTime.csv"
    download(f"{BASE}/{ds}/ExpressionData.csv",expr_p)
    download(f"{BASE}/{ds}/PseudoTime.csv",pt_p)

    pt=pd.read_csv(pt_p)
    if pt.shape[1]<2: raise ValueError(f"{ds}: malformed pseudotime")
    cell_col=pt.columns[0]; time_col=pt.columns[1]
    pt=pt[[cell_col,time_col]].copy()
    pt.columns=["cell_id","pseudotime"]
    pt["cell_id"]=pt["cell_id"].astype(str)
    pt["pseudotime"]=pd.to_numeric(pt["pseudotime"],errors="raise")
    if pt["cell_id"].duplicated().any(): raise ValueError(f"{ds}: duplicate pseudotime cell IDs")

    expr=pd.read_csv(expr_p,index_col=0)
    expr=orient_expression(expr,pt["cell_id"].tolist())
    missing_expr=[c for c in pt["cell_id"] if c not in expr.columns]
    if missing_expr: raise ValueError(f"{ds}: {len(missing_expr)} pseudotime cells absent in expression")

    # Keep exactly pseudotime trajectory cells, in pseudotime-file order.
    expr=expr[pt["cell_id"].tolist()]
    if expr.columns.tolist()!=pt["cell_id"].tolist(): raise AssertionError("cell order mismatch")

    rows=[]
    for c,t in zip(pt["cell_id"],pt["pseudotime"]):
        lookup=norm_for_labels(c)
        if lookup not in ct.index:
            raise ValueError(f"{ds}: phenotype not found for {c} -> {lookup}")
        r=ct.loc[lookup]
        macro,n_macro=macro_from_row(r)
        task_label=macro if macro in cfg["classes"] else ""
        task_id=cfg["class_order"].get(task_label,-1)
        row={
            "cell_id":c,
            "phenotype_lookup_id":lookup,
            "pseudotime":float(t),
            "phenotype_macro":macro,
            "phenotype_macro_n":int(n_macro),
            "task_label":task_label,
            "task_label_id":int(task_id),
            "task_eligible":bool(task_label),
        }
        for col in required:
            row[f"FACS_{col}"]=int(r[col])
        rows.append(row)
    meta=pd.DataFrame(rows)

    # Audit: collapsed macro labels must be mutually exclusive when present.
    if (meta["phenotype_macro"]=="AMBIGUOUS").any():
        raise ValueError(f"{ds}: ambiguous collapsed macro labels detected")

    dsout=OUT/ds; dsout.mkdir(parents=True)
    expr.to_csv(dsout/"ExpressionData.csv.gz",compression="gzip")
    meta.to_csv(dsout/"CellMetadata.csv",index=False)
    meta[["cell_id","pseudotime"]].to_csv(dsout/"PseudoTime.csv",index=False)
    meta.loc[meta.task_eligible,["cell_id","pseudotime","task_label","task_label_id"]].to_csv(
        dsout/"TaskCells.csv",index=False
    )
    # Compact machine-readable representation, no preprocessing beyond source values.
    np.savez_compressed(
        dsout/"dataset.npz",
        expression=expr.to_numpy(dtype=np.float32),
        genes=np.asarray(expr.index.astype(str),dtype=str),
        cells=np.asarray(expr.columns.astype(str),dtype=str),
        pseudotime=meta["pseudotime"].to_numpy(np.float32),
        phenotype_macro=np.asarray(meta["phenotype_macro"].astype(str),dtype=str),
        task_label=np.asarray(meta["task_label"].astype(str),dtype=str),
        task_label_id=meta["task_label_id"].to_numpy(np.int64),
    )

    all_counts=meta["phenotype_macro"].replace("", "UNLABELED").value_counts().to_dict()
    task_counts=meta.loc[meta.task_eligible,"task_label"].value_counts().to_dict()
    summary={
        "dataset":ds,
        "description":cfg["description"],
        "genes":int(expr.shape[0]),
        "trajectory_cells":int(expr.shape[1]),
        "real_phenotype_labeled_cells":int((meta.phenotype_macro!="").sum()),
        "unlabeled_cells":int((meta.phenotype_macro=="").sum()),
        "task_eligible_cells":int(meta.task_eligible.sum()),
        "task_classes":"|".join(cfg["classes"]),
        "task_counts_json":json.dumps(task_counts,sort_keys=True),
        "all_macro_counts_json":json.dumps(all_counts,sort_keys=True),
        "pseudotime_min":float(meta.pseudotime.min()),
        "pseudotime_max":float(meta.pseudotime.max()),
    }
    root_summary.append(summary)
    print("SUMMARY",json.dumps(summary,sort_keys=True))

pd.DataFrame(root_summary).to_csv(OUT/"dataset_summary.csv",index=False)
(OUT/"source_manifest.json").write_text(json.dumps(source_manifest,indent=2),encoding="utf-8")

readme="""# GSE81682 mHSC real-phenotype graph-classification datasets

This package joins three BEELINE mHSC trajectory subsets with real cell phenotype
annotations from the original Nestorowa FACS/index-sorting resources.

## Files
For each of mHSC-E, mHSC-GM and mHSC-L:
- ExpressionData.csv.gz: genes x trajectory cells, source values preserved.
- CellMetadata.csv: pseudotime, real phenotype annotations, original FACS gate flags,
  and recommended task labels.
- PseudoTime.csv: cell_id + pseudotime.
- TaskCells.csv: cells belonging to the recommended phenotype-classification task.
- dataset.npz: compact matrix/metadata copy for Python experiments.

## Real phenotype task definitions
- mHSC-E: LTHSC, MPP, CMP, MEP
- mHSC-GM: LTHSC, MPP, CMP, GMP
- mHSC-L: LTHSC, MPP, LMPP

The broader six phenotype groups are defined only by FACS annotations:
LTHSC=(LTHSC OR ESLAM OR HSC1);
MPP=(MPP OR MPP1 OR MPP2 OR MPP3 OR STHSC);
LMPP, CMP, MEP and GMP retain their corresponding gates.

## Critical anti-leakage rule
Do NOT form pseudo-samples by phenotype label.
For graph classification, sort/neighbor cells using pseudotime only, construct each
pseudo-sample, and only then inspect CellMetadata.csv to assign/accept its label.
Any window whose phenotype composition is too mixed should be excluded by a
pre-specified purity rule, not reconstructed using the labels.

## Provenance
- GEO: GSE81682, Nestorowa et al.
- FACS phenotype file: blood.stemcells.cam.ac.uk/data/cell_types.txt
- Expression and pseudotime trajectory copies: BEELINE-data in prsigma/CellProphet.
"""
(OUT/"README.md").write_text(readme,encoding="utf-8")

# Checksums
files=[p for p in OUT.rglob("*") if p.is_file()]
with (OUT/"SHA256SUMS.txt").open("w",encoding="utf-8") as f:
    for p in sorted(files):
        h=hashlib.sha256(p.read_bytes()).hexdigest()
        f.write(f"{h}  {p.relative_to(OUT).as_posix()}\n")

archive=shutil.make_archive(str(OUT),"zip",root_dir=OUT)
print("ARCHIVE",archive,Path(archive).stat().st_size)

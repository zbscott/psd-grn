#!/usr/bin/env python3
from pathlib import Path
import urllib.request, pandas as pd, numpy as np
TMP=Path("tmp_mhsc_stats"); TMP.mkdir(exist_ok=True)
urls={
"all":"https://blood.stemcells.cam.ac.uk/data/all_cell_types.txt",
"celltypes":"https://blood.stemcells.cam.ac.uk/data/cell_types.txt",
"meta":"https://blood.stemcells.cam.ac.uk/data/bloodMeta_wj.txt",
"E":"https://media.githubusercontent.com/media/prsigma/CellProphet/main/BEELINE-data/mHSC-E/PseudoTime.csv",
"GM":"https://media.githubusercontent.com/media/prsigma/CellProphet/main/BEELINE-data/mHSC-GM/PseudoTime.csv",
"L":"https://media.githubusercontent.com/media/prsigma/CellProphet/main/BEELINE-data/mHSC-L/PseudoTime.csv",
}
for n,u in urls.items():
    urllib.request.urlretrieve(u,TMP/f"{n}.txt")

def read_ws(p):
    return pd.read_csv(p, sep=r"\s+", engine="python", index_col=0)

anno=read_ws(TMP/"all.txt")
ct=read_ws(TMP/"celltypes.txt")
meta=read_ws(TMP/"meta.txt")
print("ALL",anno.shape,anno.columns.tolist())
print("CELLTYPES",ct.shape,ct.columns.tolist())
print("CELLTYPES SUMS")
for c in ct.columns:
    try: print(c,int(pd.to_numeric(ct[c],errors="coerce").fillna(0).sum()))
    except: pass
print("META",meta.shape,meta.columns.tolist())
major=["LTHSC_broad","LMPP_broad","MPP_broad","CMP_broad","MEP_broad","GMP_broad","STHSC_broad"]
sub=["MPP1_broad","MPP2_broad","MPP3_broad"]
for ds in ["E","GM","L"]:
    pt=pd.read_csv(TMP/f"{ds}.txt")
    cells=pt.iloc[:,0].astype(str)
    common=[c for c in cells if c in anno.index]
    a=anno.loc[common]
    print("\nDATASET",ds,"PT",len(cells),"HSPC_OVERLAP",len(common))
    print("BROAD COUNTS",{c:int(a[c].sum()) for c in major+sub})
    # label scheme A: collapse MPP family and keep only exactly one macro category
    z=pd.DataFrame(index=a.index)
    z["LTHSC"]=a["LTHSC_broad"].astype(int)
    z["LMPP"]=a["LMPP_broad"].astype(int)
    z["MPP"]=a[["MPP_broad","MPP1_broad","MPP2_broad","MPP3_broad"]].max(axis=1).astype(int)
    z["CMP"]=a["CMP_broad"].astype(int)
    z["MEP"]=a["MEP_broad"].astype(int)
    z["GMP"]=a["GMP_broad"].astype(int)
    z["STHSC"]=a["STHSC_broad"].astype(int)
    npos=z.sum(axis=1)
    clean=z[npos==1]
    labels=clean.idxmax(axis=1)
    print("CLEAN_UNIQUE_MACRO",len(clean),labels.value_counts().to_dict())
    print("AMBIG_MACRO",int((npos>1).sum()),"UNLABELED",int((npos==0).sum()))
    # scheme B: exact/narrow columns from all_cell_types, major
    exact_cols=["LTHSC","LMPP","MPP","CMP","MEP","GMP","STHSC","ESLAM"]
    e=a[exact_cols].astype(int)
    en=e.sum(axis=1)
    ec=e[en==1].idxmax(axis=1)
    print("CLEAN_EXACT",int((en==1).sum()),ec.value_counts().to_dict(),"AMBIG",int((en>1).sum()),"NONE",int((en==0).sum()))
    # cell_types overlap and sums
    cc=[c for c in cells if c.replace("_","-") in ct.index or c in ct.index]
    print("CELLTYPES_OVERLAP_RAW",len(cc))

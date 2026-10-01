#!/usr/bin/env python3
from pathlib import Path
import urllib.request, pandas as pd, numpy as np
TMP=Path("tmp_mhsc_ctstats"); TMP.mkdir(exist_ok=True)
urls={
"ct":"https://blood.stemcells.cam.ac.uk/data/cell_types.txt",
"E":"https://media.githubusercontent.com/media/prsigma/CellProphet/main/BEELINE-data/mHSC-E/PseudoTime.csv",
"GM":"https://media.githubusercontent.com/media/prsigma/CellProphet/main/BEELINE-data/mHSC-GM/PseudoTime.csv",
"L":"https://media.githubusercontent.com/media/prsigma/CellProphet/main/BEELINE-data/mHSC-L/PseudoTime.csv",
}
for n,u in urls.items(): urllib.request.urlretrieve(u,TMP/f"{n}.txt")
ct=pd.read_csv(TMP/"ct.txt",sep=r"\s+",engine="python",index_col=0)
ct=ct.astype(str).apply(lambda s:s.str.upper().map({"TRUE":1,"FALSE":0}).fillna(pd.to_numeric(s,errors="coerce"))).fillna(0).astype(int)
print("CT_SHAPE",ct.shape)
print("CT_COLS",ct.columns.tolist())
print("CT_SUMS",{c:int(ct[c].sum()) for c in ct.columns})
npos=ct.sum(axis=1)
print("CT_ROW_POS_COUNTS",npos.value_counts().sort_index().to_dict())

def norm_cell(c):
    c=str(c)
    if c.startswith("LT_HSC_"): return "LT-HSC_"+c[len("LT_HSC_"):]
    return c

for ds in ["E","GM","L"]:
    pt=pd.read_csv(TMP/f"{ds}.txt")
    raw=pt.iloc[:,0].astype(str)
    norm=raw.map(norm_cell)
    matched=norm.isin(ct.index)
    print("\nDATASET",ds,"N",len(raw),"MATCHED",int(matched.sum()),"UNMATCHED",int((~matched).sum()))
    if (~matched).sum():
        print("UNMATCHED_SAMPLE",raw[~matched].head(30).tolist())
    sub=ct.loc[norm[matched]]
    sub.index=raw[matched].values
    n=sub.sum(axis=1)
    print("ONEHOT",int((n==1).sum()),"MULTI",int((n>1).sum()),"NONE",int((n==0).sum()))
    one=sub[n==1]
    labs=one.idxmax(axis=1)
    print("ONEHOT_LABELS",labs.value_counts().to_dict())
    # collapsed 6-class phenotype:
    z=pd.DataFrame(index=sub.index)
    z["LTHSC"]=sub[["LTHSC","ESLAM","HSC1"]].max(axis=1)
    z["LMPP"]=sub[["LMPP"]].max(axis=1)
    z["MPP"]=sub[["MPP","MPP1","MPP2","MPP3","STHSC"]].max(axis=1)
    z["CMP"]=sub[["CMP"]].max(axis=1)
    z["MEP"]=sub[["MEP"]].max(axis=1)
    z["GMP"]=sub[["GMP"]].max(axis=1)
    zn=z.sum(axis=1)
    clean=z[zn==1]
    lab=clean.idxmax(axis=1)
    print("COLLAPSED6_CLEAN",len(clean),"MULTI",int((zn>1).sum()),"NONE",int((zn==0).sum()),"COUNTS",lab.value_counts().to_dict())

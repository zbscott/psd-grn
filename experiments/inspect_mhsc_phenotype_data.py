#!/usr/bin/env python3
from pathlib import Path
import urllib.request
import pandas as pd
import re, json

ROOT=Path(__file__).resolve().parents[1]
TMP=ROOT/"tmp_mhsc_build"
TMP.mkdir(exist_ok=True)

URLS={
"all_cell_types.txt":"https://blood.stemcells.cam.ac.uk/data/all_cell_types.txt",
"cell_name_conversion.csv":"https://blood.stemcells.cam.ac.uk/data/cell_name_conversion.csv",
"cell_names_nestorowa_data.txt":"https://blood.stemcells.cam.ac.uk/data/cell_names_nestorowa_data.txt",
"mHSC-E_PseudoTime.csv":"https://media.githubusercontent.com/media/prsigma/CellProphet/main/BEELINE-data/mHSC-E/PseudoTime.csv",
"mHSC-GM_PseudoTime.csv":"https://media.githubusercontent.com/media/prsigma/CellProphet/main/BEELINE-data/mHSC-GM/PseudoTime.csv",
"mHSC-L_PseudoTime.csv":"https://media.githubusercontent.com/media/prsigma/CellProphet/main/BEELINE-data/mHSC-L/PseudoTime.csv",
}

for name,url in URLS.items():
    p=TMP/name
    urllib.request.urlretrieve(url,p)
    print("DOWNLOADED",name,p.stat().st_size)

for name in ["all_cell_types.txt","cell_name_conversion.csv","cell_names_nestorowa_data.txt"]:
    p=TMP/name
    print("\nFILE",name)
    print(p.read_text(errors="replace")[:3000])

def read_pt(p):
    df=pd.read_csv(p)
    print("COLUMNS",df.columns.tolist(),"SHAPE",df.shape)
    print(df.head(10).to_string(index=False))
    return df

pts={}
for ds in ["mHSC-E","mHSC-GM","mHSC-L"]:
    print("\nPT",ds)
    pts[ds]=read_pt(TMP/f"{ds}_PseudoTime.csv")

anno=pd.read_csv(TMP/"all_cell_types.txt", sep=r"\s+", engine="python", index_col=0)
print("\nANNO SHAPE",anno.shape)
print("ANNO INDEX SAMPLE",anno.index[:20].tolist())
print("ANNO COLS",anno.columns.tolist())

conv=pd.read_csv(TMP/"cell_name_conversion.csv")
print("\nCONV SHAPE",conv.shape)
print(conv.head(20).to_string(index=False))

for ds,df in pts.items():
    cells=df.iloc[:,0].astype(str)
    print("\nOVERLAP",ds,"n",len(cells))
    print("PT SAMPLE",cells.head(20).tolist())
    direct=sum(c in set(anno.index.astype(str)) for c in cells)
    norm=lambda s: str(s).replace("-",".")
    anno_norm=set(map(norm,anno.index.astype(str)))
    normalized=sum(norm(c) in anno_norm for c in cells)
    print("DIRECT",direct,"NORMALIZED",normalized)

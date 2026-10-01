#!/usr/bin/env python3
from pathlib import Path
import urllib.request, pandas as pd
TMP=Path("tmp_mhsc_labels2"); TMP.mkdir(exist_ok=True)
urls={
"cell_types.txt":"https://blood.stemcells.cam.ac.uk/data/cell_types.txt",
"nestorowa_corrected_population_annotation.txt":"https://blood.stemcells.cam.ac.uk/data/nestorowa_corrected_population_annotation.txt",
"bloodMeta_wj.txt":"https://blood.stemcells.cam.ac.uk/data/bloodMeta_wj.txt",
"cluster_ids.txt":"https://blood.stemcells.cam.ac.uk/data/cluster_ids.txt",
"diffusionMapCoords.txt":"https://blood.stemcells.cam.ac.uk/data/diffusionMapCoords.txt",
}
for n,u in urls.items():
    p=TMP/n; urllib.request.urlretrieve(u,p)
    print("\n###",n,p.stat().st_size)
    print(p.read_text(errors="replace")[:8000])

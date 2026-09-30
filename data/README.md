# Data

Large public datasets are not duplicated in this repository. Download them from their original sources and place the processed inputs in the formats below.

## Benchmark data

The benchmark uses the BEELINE single-cell datasets under TF+500 and TF+1000 gene-set settings. A TF+1000 hESC example is included in `data/demo/hESC/`.

Expected dataset directory:

```text
Datasets/
└── hESC/
    ├── ExpressionData1000.csv
    ├── PseudoTime.csv
    └── gold_augmented.csv
```

The same layout is used for the remaining datasets.

`ExpressionData*.csv` contains the gene-expression matrix, `PseudoTime.csv` contains cell pseudotime values, and `gold_augmented.csv` contains the signed directed reference relationships used by the manuscript pipeline.

## IPF discovery and replication data

Public cohorts:

- GSE135893: discovery cohort;
- GSE136831: independent IPF replication cohort;
- GSE249973: RBPJ/SHARP chromatin analysis;
- GSE284531: SOX9 perturbation analysis.

Trajectory-level model inputs are not stored because the single-cell preprocessing files are large. The downstream scripts expect processed expression, pseudotime, regulatory-reference, probability, and locus-level tables as documented in `docs/data_formats.md`.

## Regulatory-state terminology

For the signed three-class task:

```text
0  No recorded edge
1  Activation
2  Repression
```

`No recorded edge` denotes absence from the reference regulatory network used in the analysis; it does not assert biological absence of regulation.

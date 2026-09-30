# Downstream data formats

## Trajectory expression matrix

CSV with genes as rows and cells as columns. The first column contains gene identifiers. Repair-associated and disease-associated trajectories must use the same gene-node space for direct ordered-pair comparison.

## Pseudotime table

```text
cell,pseudotime
AAAC...,0.0123
AAAG...,0.0198
```

## Signed reference network

```text
source,target,sign
GENEA,GENEB,1
GENEC,GENED,-1
```

Positive sign denotes Activation and negative sign denotes Repression.

## Discovery all-pair consensus table

Produced by `scripts/ipf/infer_trajectory.py --mode all`:

```text
source,target,consensus_state,consensus_votes,
mean_p_no_edge,mean_p_activation,mean_p_repression
```

A discovery state is stable when `consensus_votes >= 4` across five models.

## Frozen-candidate probability table

Produced by `scripts/ipf/infer_trajectory.py --mode candidates`:

```text
source,target,split,p_no_edge,p_activation,p_repression
```

## Donor-balanced repeat-state table

```text
source,target,repeat,repair_state,disease_state
SPEN,SOX9,0,No recorded edge,Activation
```

Allowed states are exactly `No recorded edge`, `Activation`, and `Repression`.

## RBPJ locus table

```text
gene,is_spen_candidate,promoter_peak,cis_peak,wt_signal,ko30_signal,ko36_signal
```

The manuscript definitions are:

- promoter occupancy: WT RBPJ peak within +/-2 kb of the transcription start site;
- cis occupancy: WT RBPJ peak within +/-100 kb;
- joint KO/WT ratio: mean of KO30 and KO36 locus signal divided by WT locus signal.

## GSE284531 workbook

The GEO supplementary workbook `GSE284531_raw_counts_All_Sample.xlsx` contains a worksheet named `0.05 FDR DEG` with author-provided `logFC`, `FDR`, and sample-level values. The SOX9 analysis reads these reported statistics directly and does not recompute genome-wide differential expression.

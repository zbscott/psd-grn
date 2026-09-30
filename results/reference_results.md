# Manuscript reference checkpoints

These values are included only as reproducibility sanity checks. The analysis scripts do not hard-code them.

## Benchmark / stage sensitivity

- Signed directed benchmark uses six datasets after excluding mHSC-L from the three-class task.
- Formal stage sensitivity uses `T = 3,4,5,6,7` with all other settings fixed.
- `T = 5` is the retained default stage resolution.

## GSE135893 discovery

- Common node space: 1,525 genes.
- Ordered pairs of distinct genes: 2,324,100.
- Recorded reference relationships: 1,482.
- Unrecorded ordered pairs: 2,322,618.
- Stable unrecorded rewiring pairs satisfying >=4/5 support in both trajectories: 5,365.
- Source-route candidates: 430.
- Marker-target candidates: 54.
- Overlap between routes: 20.
- Frozen prioritized discovery candidates: 464.
- Rewiring composition: 286 disease-associated gains, 155 losses, 23 sign changes.
- Donor-balanced robust candidates: 49/464 (>=4/5 paired resamplings).

## GSE136831 independent replication

- Frozen discovery candidates: 464.
- Externally evaluable candidates: 463 (TAF9->KLF4 unavailable because one node could not be mapped).
- Primary exact transition replications: 58/463 (12.5%).
- Global empirical null mean: approximately 27.5 exact matches; manuscript empirical P = 1.0e-5 with +1 correction over 100,000 permutations.
- Source-preserving permutation sensitivity: manuscript empirical P = 0.0028.
- Disease-associated gains: 47/285 exact externally replicated.
- Disease-associated losses: 6/155 exact externally replicated.

## Source-level result

Across 128 regulatory-source tests:

- SPEN: 8/10 exact replications.
- Other sources: 50/453 exact replications.
- one-sided Fisher odds ratio: 32.24.
- P = 1.42e-6.
- BH q = 1.82e-4.

SPEN was the only regulatory source with BH q < 0.05 in the all-candidate exploratory source-level analysis.

## Orthogonal analyses

RBPJ/SHARP context:

- promoter (±2 kb) WT RBPJ occupancy: 8/10 SPEN candidate targets;
- cis (±100 kb) WT RBPJ occupancy: 10/10;
- candidate-background frequencies: 137/185 and 172/185, respectively;
- Fisher P = 0.504 and 0.493;
- median joint KO/WT ratio: candidate loci 0.7185, background 0.6855;
- one-sided rank-sum P = 0.594.

SOX9 perturbation (GSE284531 author-provided filtered DEG table):

- 1,762 genes with FDR < 0.05;
- 823 decreased, 939 increased after SOX9 knockdown;
- SOX9 logFC = -1.521 (rounded), FDR = 1.08e-13;
- 5/20 predefined aberrant-basaloid-associated genes are evaluable in the filtered table, with changes in both directions.

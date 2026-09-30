# PSD-GRN

PSD-GRN is a pseudotime-resolved signed directed graph neural network for gene regulatory relationship prediction from single-cell RNA-seq data. The repository contains the core model implementation used for the benchmark experiments together with parameterized scripts for the trajectory-specific regulatory rewiring analyses reported in the manuscript.

## Repository structure

```text
PSD-GRN/
├── README.md
├── requirements.txt
├── src/                         # core PSD-GRN model and training code
├── scripts/
│   ├── benchmark/               # benchmark and stage-sensitivity runners
│   ├── ipf/                     # trajectory-specific rewiring analyses
│   └── orthogonal/              # RBPJ/SHARP and SOX9 analyses
├── configs/                     # example path/layout configurations
├── data/
│   ├── README.md                # data sources and expected formats
│   └── demo/hESC/               # small BEELINE-format demo dataset
├── docs/
│   ├── ablation.md              # exact definitions of the four ablation variants
│   └── data_formats.md          # input/output tables used by downstream analyses
└── results/
    └── reference_results.md     # manuscript-level reference values for checking runs
```

## Installation

Python 3.10 or later is recommended. A CUDA-capable GPU is recommended for model training.

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

PyTorch and PyTorch Geometric wheels depend on the local CUDA version. If necessary, install a CUDA-compatible PyTorch/PyG build first and then install the remaining dependencies.

## Core benchmark

The benchmark code follows the BEELINE-style single-cell GRN setting used in the manuscript. The two prediction tasks are:

- directed regulatory edge existence: `No recorded edge` vs `Edge exists`;
- signed directed regulatory relationship prediction: `No recorded edge`, `Activation`, and `Repression`.

The signed three-class labels are encoded as:

```text
0  No recorded edge
1  Activation
2  Repression
```

A TF+1000 hESC demo dataset is included under `data/demo/hESC/`.

A short environment smoke test can be run with one split and one epoch (this is only a runtime check, not a manuscript result):

```bash
python src/train_psdgrn_3c.py \
  --dataset hESC \
  --expression_file ExpressionData1000.csv \
  --data_root data/demo \
  --output_root outputs/smoke \
  --num_classes 3 \
  --K 1 --q 0.1 --hidden 64 --bins 5 \
  --lr 0.001 --weight_decay 0.05 --dropout 0.5 \
  --epochs 1 --patience 1 --checkpoint 1 \
  --train_ratio 0.8 --val_ratio 0.1 \
  --runs 1 --seed 0 --method PSD-GRN
```

For the article-configuration benchmark:

```bash
python scripts/benchmark/run_benchmark.py \
  --task signed \
  --gene-set 1000 \
  --datasets hESC \
  --data-root data/demo \
  --output-root outputs
```

For the full benchmark, place the BEELINE-format datasets under a local data directory following the layout in `data/README.md` and `configs/benchmark_data_layout.txt`.

## Pseudotime-stage sensitivity

The manuscript evaluates `T = 3, 4, 5, 6, 7` consecutive pseudotime stages while holding all other training settings fixed.

```bash
python scripts/benchmark/run_stage_sensitivity.py \
  --data-root /path/to/Datasets \
  --output-root outputs/stage_sensitivity

python scripts/benchmark/summarize_stage_sensitivity.py \
  --root outputs/stage_sensitivity
```

`T = 5` is the default stage resolution used in the main experiments.

## Trajectory-specific regulatory rewiring

The IPF analysis compares the same ordered gene pair between a repair-associated epithelial trajectory and a disease-associated epithelial trajectory. The regulatory state of each ordered pair is represented as `No recorded edge`, `Activation`, or `Repression`.

The analysis workflow is:

1. train five PSD-GRN models for each trajectory using matched graph splits;
2. infer the three-class state of all ordered pairs;
3. identify stable cross-trajectory state changes;
4. prioritize the frozen discovery candidate set;
5. evaluate donor-balanced robustness;
6. retrain PSD-GRN in the independent GSE136831 cohort and test exact two-trajectory replication;
7. perform source-level enrichment and orthogonal analyses.

### Discovery cohort

After training the repair-associated and disease-associated models, infer all ordered pairs:

```bash
python scripts/ipf/infer_trajectory.py \
  --dataset GSE135893_repair \
  --expression /path/to/repair_expression.csv \
  --pseudotime /path/to/repair_pseudotime.csv \
  --gold /path/to/reference_signed.csv \
  --output-root outputs/GSE135893_repair \
  --mode all \
  --out outputs/GSE135893_repair/all_pairs_consensus.csv
```

Run the same command for the disease-associated trajectory and then prioritize cross-trajectory rewiring candidates:

```bash
python scripts/ipf/discovery_rewiring.py \
  --repair outputs/GSE135893_repair/all_pairs_consensus.csv \
  --disease outputs/GSE135893_disease/all_pairs_consensus.csv \
  --reference /path/to/reference_signed.csv \
  --regulators /path/to/BEELINE_regulators.csv \
  --outdir outputs/discovery_rewiring
```

The discovery analysis uses at least 4/5 model agreement in both trajectories, the six repair-to-disease regulatory-state transitions, and the ranking score described in the manuscript.

### Donor-balanced robustness

```bash
python scripts/ipf/donor_balanced_robustness.py \
  --discovery outputs/discovery_rewiring/prioritized_discovery_candidates.csv \
  --repeat-states /path/to/donor_balanced_repeat_states.csv \
  --out outputs/donor_balanced_candidates.csv
```

### Independent replication in GSE136831

Infer the frozen discovery candidates in five independently trained external models for each trajectory, then evaluate exact replication:

```bash
python scripts/ipf/external_replication.py \
  --discovery outputs/discovery_rewiring/prioritized_discovery_candidates.csv \
  --repair outputs/GSE136831_repair/frozen_candidate_probabilities.csv \
  --disease outputs/GSE136831_disease/frozen_candidate_probabilities.csv \
  --out outputs/GSE136831_external_replication.csv \
  --n-perm 100000
```

The primary external endpoint uses mean class probabilities across the five external models within each trajectory. Exact replication requires both external trajectory states to match the frozen discovery states.

### Source-level enrichment

```bash
python scripts/ipf/source_enrichment.py \
  --replication outputs/GSE136831_external_replication.csv \
  --outdir outputs/source_enrichment
```

### Expression-profile divergence

```bash
python scripts/ipf/expression_divergence.py \
  --repair-expression /path/to/repair_expression.csv \
  --repair-pseudotime /path/to/repair_pseudotime.csv \
  --disease-expression /path/to/disease_expression.csv \
  --disease-pseudotime /path/to/disease_pseudotime.csv \
  --candidates outputs/discovery_rewiring/prioritized_discovery_candidates.csv \
  --external-replication outputs/GSE136831_external_replication.csv \
  --out outputs/expression_divergence.csv
```

## Orthogonal analyses

### RBPJ/SHARP chromatin context

```bash
python scripts/orthogonal/rbpj_chromatin.py \
  --loci /path/to/rbpj_locus_table.csv \
  --outdir outputs/rbpj
```

### SOX9 perturbation

```bash
python scripts/orthogonal/sox9_perturbation.py \
  --workbook /path/to/GSE284531_raw_counts_All_Sample.xlsx \
  --outdir outputs/sox9
```

## Ablation experiments

The four ablations were run by modifying the same three-class training pipeline used for the full PSD-GRN model. The exact experimental definitions are provided in `docs/ablation.md`. The intermediate modified copies used for those runs were not saved.

## Data

The manuscript uses public datasets including the BEELINE benchmark datasets and GEO accessions GSE135893, GSE136831, GSE249973, and GSE284531. Large public raw data and model checkpoints are not stored in this repository. See `data/README.md` for the expected local file organization.

## Reference values

`results/reference_results.md` lists manuscript-level values that can be used to check a completed analysis, including the discovery candidate count, independent-cohort replication count, and source-level SPEN result. These values are not hard-coded into the analysis scripts.

## Citation

Citation information will be updated after publication.

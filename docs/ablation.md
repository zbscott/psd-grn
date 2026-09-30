# Ablation experiments

The ablation experiments were performed for the TF+1000 signed directed three-class task across the six datasets used in the manuscript. All variants used the same data splits and candidate-sampling rules as the full PSD-GRN model.

The ablations were produced by modifying the main three-class training pipeline for each experiment. The intermediate modified copies used for those runs were not saved. The exact experimental definitions used in the manuscript are listed below.

## w/o SR

Replace the inter-stage residual connections with concatenation of representations from different pseudotime stages. All other model settings remain unchanged.

## w/o PT

Randomly reassign cell expression profiles across the fixed pseudotime positions using one shared permutation. The original pseudotime values and stage partitioning are retained. This breaks the correspondence between expression profiles and pseudotime while leaving the remaining pipeline unchanged.

## w/o PEF

Replace the pseudotime-stage feature construction with an MLP that maps raw expression features to node features whose output dimension matches the input dimension of the convolution module. The subsequent signed directed magnetic Laplacian convolution, hidden dimension, and classifier remain unchanged.

## w/o MLC

Replace the signed directed magnetic Laplacian convolution with a standard GCN. This removes explicit encoding of regulatory direction and Activation/Repression sign from the convolution operator.

For Figure 3, performance is first averaged across five repeated experiments within each dataset and then summarized across the six dataset-level means.

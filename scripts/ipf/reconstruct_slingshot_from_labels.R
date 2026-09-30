#!/usr/bin/env Rscript
# Clean, parameterized trajectory reconstruction helper.
#
# IMPORTANT PROVENANCE NOTE:
# This helper is not claimed to be the byte-identical historical Phase 4E R runner.
# It exposes the manuscript-level operation (Slingshot fitted separately to a
# pre-specified start/end state) once cell-state labels and a low-dimensional
# embedding have already been prepared.

suppressPackageStartupMessages({
  library(SingleCellExperiment)
  library(slingshot)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 6) {
  stop(paste(
    "Usage: Rscript reconstruct_slingshot_from_labels.R",
    "<embedding.csv> <metadata.csv> <cell_col> <state_col>",
    "<start_state> <end_state> [output.csv]"
  ))
}

embedding_file <- args[[1]]
metadata_file <- args[[2]]
cell_col <- args[[3]]
state_col <- args[[4]]
start_state <- args[[5]]
end_state <- args[[6]]
out_file <- ifelse(length(args) >= 7, args[[7]], "trajectory_pseudotime.csv")

emb <- read.csv(embedding_file, check.names = FALSE, row.names = 1)
meta <- read.csv(metadata_file, check.names = FALSE, stringsAsFactors = FALSE)
if (!all(c(cell_col, state_col) %in% colnames(meta))) {
  stop("metadata is missing requested cell/state columns")
}
meta <- meta[meta[[state_col]] %in% c(start_state, end_state), , drop = FALSE]
common <- intersect(rownames(emb), meta[[cell_col]])
if (length(common) < 3) stop("Too few shared trajectory cells")
emb <- as.matrix(emb[common, , drop = FALSE])
meta <- meta[match(common, meta[[cell_col]]), , drop = FALSE]
labels <- factor(meta[[state_col]], levels = c(start_state, end_state))

sce <- SingleCellExperiment(assays = list(dummy = matrix(0, nrow = 1, ncol = length(common))))
colnames(sce) <- common
reducedDim(sce, "embedding") <- emb
colData(sce)$state <- labels
sce <- slingshot(sce, clusterLabels = "state", reducedDim = "embedding",
                 start.clus = start_state, end.clus = end_state)
pt <- slingPseudotime(sce)
if (ncol(pt) < 1) stop("Slingshot returned no lineage pseudotime")
p <- pt[, 1]
out <- data.frame(cell = names(p), pseudotime = as.numeric(p), state = as.character(labels))
out <- out[is.finite(out$pseudotime), , drop = FALSE]
write.csv(out, out_file, row.names = FALSE)
cat(sprintf("Wrote %d cells to %s\n", nrow(out), out_file))

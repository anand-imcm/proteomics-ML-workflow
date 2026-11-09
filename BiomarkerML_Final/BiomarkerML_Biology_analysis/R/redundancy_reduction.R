jaccard_similarity <- function(x, y) {
  union_n <- length(union(x, y)); if (!union_n) return(0)
  length(intersect(x, y)) / union_n
}

reduce_pathway_redundancy <- function(enrichment, threshold = 0.5) {
  if (!nrow(enrichment)) return(enrichment)
  sets <- strsplit(enrichment$SupportingProteins, ";", fixed = TRUE)
  n <- length(sets)
  sim <- diag(n)
  if (n > 1) for (i in seq_len(n - 1)) for (j in (i + 1):n) sim[i, j] <- sim[j, i] <- jaccard_similarity(sets[[i]], sets[[j]])
  groups <- if (n == 1) 1L else stats::cutree(stats::hclust(stats::as.dist(1 - sim), method = "complete"), h = 1 - threshold)
  enrichment$RedundancyCluster <- as.integer(groups)
  # Lowest FDR, then largest support, then stable lexical term ID decides each representative.
  enrichment |>
    dplyr::arrange(.data$RedundancyCluster, .data$AdjustedPValue, dplyr::desc(.data$GeneCount), .data$TermID) |>
    dplyr::group_by(.data$InputSet, .data$Database, .data$RedundancyCluster) |>
    dplyr::slice(1) |>
    dplyr::ungroup()
}

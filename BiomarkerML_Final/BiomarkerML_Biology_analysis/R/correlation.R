expression_wide <- function(expression, proteins) {
  x <- dplyr::filter(expression, .data$Protein %in% proteins)
  wide <- tidyr::pivot_wider(x, id_cols = "SampleID", names_from = "Protein", values_from = "Abundance")
  out <- as.data.frame(wide[setdiff(names(wide), "SampleID")])
  rownames(out) <- wide$SampleID
  out
}

validate_expression_groups <- function(expression, config) {
  if (!"Group" %in% names(expression) || all(is.na(expression$Group)) || !any(nzchar(expression$Group[!is.na(expression$Group)])))
    stopf("Disease-specific correlation requires sample phenotype metadata. Configure project.expression_group_column and supply PD/ALS group labels in expression_matrix.csv.")
  expected <- c(config$correlation$pd_label, config$correlation$als_label)
  missing <- setdiff(expected, unique(expression$Group))
  if (length(missing)) stopf("Expression phenotype column does not contain configured disease label(s): %s", paste(missing, collapse = ", "))
  invisible(TRUE)
}

filter_expression_cohort <- function(expression, group_label, config) {
  if (!isTRUE(config$correlation$disease_samples_only))
    stopf("Primary Figure X-C requires correlation.disease_samples_only: true")
  validate_expression_groups(expression, config)
  out <- dplyr::filter(expression, .data$Group == group_label)
  if (dplyr::n_distinct(out$SampleID) < 3) stopf("At least three '%s' samples are required for Spearman correlation", group_label)
  out
}

calculate_spearman_correlation <- function(expression, proteins) {
  present <- intersect(proteins, unique(expression$Protein))
  missing <- setdiff(proteins, present)
  if (length(present) < 2) return(list(correlation = matrix(numeric(), 0, 0), pvalue = matrix(numeric(), 0, 0), n = 0L, missing = missing))
  mat <- expression_wide(expression, present)
  cor_mat <- stats::cor(mat, method = "spearman", use = "pairwise.complete.obs")
  n <- nrow(mat)
  p_mat <- matrix(NA_real_, ncol(mat), ncol(mat), dimnames = list(colnames(mat), colnames(mat)))
  for (i in seq_len(ncol(mat))) for (j in seq_len(ncol(mat))) if (i != j) {
    ok <- stats::complete.cases(mat[, i], mat[, j])
    if (sum(ok) >= 3) p_mat[i, j] <- stats::cor.test(mat[ok, i], mat[ok, j], method = "spearman", exact = FALSE)$p.value
  }
  diag(p_mat) <- 0
  adjusted <- matrix(NA_real_, ncol(mat), ncol(mat), dimnames = dimnames(p_mat))
  if (ncol(mat) > 1) {
    upper <- upper.tri(p_mat)
    adjusted[upper] <- stats::p.adjust(p_mat[upper], method = "BH")
    adjusted[lower.tri(adjusted)] <- t(adjusted)[lower.tri(adjusted)]
  }
  diag(adjusted) <- 0
  missing_fraction <- colMeans(is.na(mat))
  list(correlation = cor_mat, pvalue = p_mat, adjusted_pvalue = adjusted, n = n,
       biomarkers = ncol(mat), missing = missing, missing_fraction = missing_fraction)
}

calculate_disease_correlations <- function(expression, included, config) {
  validate_expression_groups(expression, config)
  labels <- c(PD = config$correlation$pd_label, ALS = config$correlation$als_label)
  out <- lapply(names(labels), function(analysis) {
    cohort <- filter_expression_cohort(expression, labels[[analysis]], config)
    result <- calculate_spearman_correlation(cohort, included[[analysis]]$Protein)
    result$group_label <- labels[[analysis]]
    result
  })
  names(out) <- names(labels)
  out
}

correlation_source_table <- function(result, analysis) {
  if (!length(result$correlation)) return(data.frame())
  rho <- as.data.frame(as.table(result$correlation), stringsAsFactors = FALSE)
  pvalue <- as.data.frame(as.table(result$pvalue), stringsAsFactors = FALSE)
  adjusted <- as.data.frame(as.table(result$adjusted_pvalue), stringsAsFactors = FALSE)
  data.frame(Protein1 = rho$Var1, Protein2 = rho$Var2, SpearmanRho = rho$Freq,
             PValue = pvalue$Freq, AdjustedPValue = adjusted$Freq) |>
    dplyr::mutate(Analysis = analysis, Samples = result$n) |>
    dplyr::filter(.data$Protein1 < .data$Protein2)
}

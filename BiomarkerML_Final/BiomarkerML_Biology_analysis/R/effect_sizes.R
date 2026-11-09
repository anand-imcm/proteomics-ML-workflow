validate_effect_size_config <- function(expression, config) {
  if (!identical(config$effect_size$method, "hedges_g")) stopf("Only effect_size.method: hedges_g is supported")
  labels <- c(
    PD_case = config$effect_size$pd_case_label,
    PD_control = config$effect_size$pd_control_label,
    ALS_case = config$effect_size$als_case_label,
    ALS_control = config$effect_size$als_control_label
  )
  if (anyNA(labels) || any(!nzchar(labels))) stopf("All effect-size group labels must be configured")
  present <- unique(expression$Group[!is.na(expression$Group) & nzchar(expression$Group)])
  missing <- setdiff(unique(labels), present)
  if (length(missing)) stopf("Expression phenotype column does not contain configured effect-size label(s): %s", paste(missing, collapse = ", "))
  if (identical(labels[["PD_case"]], labels[["PD_control"]]) || identical(labels[["ALS_case"]], labels[["ALS_control"]]))
    stopf("Disease and control labels must differ within each effect-size contrast")
  invisible(labels)
}

hedges_g_summary <- function(disease, control, minimum_group_n = 3L, calculate_ci = TRUE, ci_level = .95) {
  disease <- disease[is.finite(disease)]; control <- control[is.finite(control)]
  n1 <- length(disease); n0 <- length(control)
  base <- list(
    EffectStatus = "estimable", HedgesG = NA_real_, HedgesG_Variance = NA_real_, HedgesG_SE = NA_real_,
    CI_Lower = NA_real_, CI_Upper = NA_real_, N_Disease = n1, N_Control = n0,
    DiseaseMean = if (n1) mean(disease) else NA_real_, ControlMean = if (n0) mean(control) else NA_real_,
    DiseaseMedian = if (n1) stats::median(disease) else NA_real_, ControlMedian = if (n0) stats::median(control) else NA_real_,
    MeanDifference = if (n1 && n0) mean(disease) - mean(control) else NA_real_,
    MedianDifference = if (n1 && n0) stats::median(disease) - stats::median(control) else NA_real_
  )
  if (n1 < minimum_group_n || n0 < minimum_group_n) {
    base$EffectStatus <- sprintf("not_estimable: minimum_group_n_%d_not_met", minimum_group_n)
    return(base)
  }
  df <- n1 + n0 - 2
  if (df <= 0) { base$EffectStatus <- "not_estimable: nonpositive_degrees_of_freedom"; return(base) }
  # Independent-groups standardized mean difference (Hedges & Olkin form):
  # s_p^2 = ((n1 - 1)s1^2 + (n0 - 1)s0^2) / df;
  # d = (mean_disease - mean_control) / s_p;
  # J = 1 - 3 / (4*df - 1); g = J*d.
  v1 <- stats::var(disease); v0 <- stats::var(control)
  pooled_variance <- ((n1 - 1) * v1 + (n0 - 1) * v0) / df
  pooled_sd <- sqrt(pooled_variance)
  if (!is.finite(pooled_sd) || pooled_sd <= 0) {
    base$EffectStatus <- "not_estimable: pooled_sd_zero_or_nonfinite"
    return(base)
  }
  cohen_d <- base$MeanDifference / pooled_sd
  correction_j <- 1 - 3 / (4 * df - 1)
  g <- correction_j * cohen_d
  # Var(g) = J^2 * Var(d), where the independent-groups approximation is
  # Var(d) = (n1 + n0)/(n1*n0) + d^2/(2*df). Equivalently,
  # Var(g) = J^2*(n1+n0)/(n1*n0) + g^2/(2*df).
  # SE(g) = sqrt(Var(g)); the configured normal CI is g +/- z*SE(g).
  variance_g <- correction_j^2 * ((n1 + n0) / (n1 * n0) + cohen_d^2 / (2 * df))
  se_g <- sqrt(variance_g)
  if (!all(is.finite(c(g, variance_g, se_g)))) {
    base$EffectStatus <- "not_estimable: nonfinite_effect_or_variance"
    return(base)
  }
  base$HedgesG <- g; base$HedgesG_Variance <- variance_g; base$HedgesG_SE <- se_g
  if (isTRUE(calculate_ci)) {
    alpha <- 1 - ci_level
    z <- stats::qnorm(1 - alpha / 2)
    base$CI_Lower <- g - z * se_g; base$CI_Upper <- g + z * se_g
  }
  base
}

prefix_effect_summary <- function(x, prefix) {
  names(x) <- ifelse(names(x) == "EffectStatus", paste0("EffectStatus_", prefix),
                     paste0(prefix, "_", names(x)))
  as.data.frame(x, stringsAsFactors = FALSE)
}

calculate_all_protein_effect_sizes <- function(expression, measured_proteins, config) {
  validate_effect_size_config(expression, config)
  es <- config$effect_size
  proteins <- sort(unique(as.character(measured_proteins)))
  rows <- lapply(proteins, function(protein) {
    x <- dplyr::filter(expression, .data$Protein == protein)
    pd <- hedges_g_summary(
      x$Abundance[x$Group == es$pd_case_label], x$Abundance[x$Group == es$pd_control_label],
      es$minimum_group_n, es$calculate_ci, es$ci_level
    )
    als <- hedges_g_summary(
      x$Abundance[x$Group == es$als_case_label], x$Abundance[x$Group == es$als_control_label],
      es$minimum_group_n, es$calculate_ci, es$ci_level
    )
    cbind(data.frame(Protein = protein, stringsAsFactors = FALSE),
          prefix_effect_summary(pd, "PD"), prefix_effect_summary(als, "ALS"))
  })
  out <- dplyr::bind_rows(rows)
  numeric_effect_cols <- setdiff(names(out), c("Protein", "EffectStatus_PD", "EffectStatus_ALS"))
  for (nm in numeric_effect_cols) out[[nm]][!is.finite(out[[nm]])] <- NA_real_
  out
}

recommended_effect_sizes <- function(effect_sizes, included) {
  union_proteins <- sort(unique(unlist(lapply(included, `[[`, "Protein"))))
  dplyr::filter(effect_sizes, .data$Protein %in% union_proteins)
}

effect_size_method_note <- function(config) {
  sprintf(
    paste0("For independent groups, pooled variance = ((n_disease-1)*s_disease^2 + (n_control-1)*s_control^2)/df; ",
           "Cohen's d = (mean_disease-mean_control)/pooled_SD; J = 1 - 3/(4*df - 1); Hedges' g = J*d, ",
           "where df = n_disease + n_control - 2. Sampling variance is J^2*((n_disease+n_control)/(n_disease*n_control) + d^2/(2*df)); ",
           "SE(g) = sqrt(variance). ",
           "The %.1f%% CI is g +/- z_(1-alpha/2)*SE. Non-estimable effects remain NA with an explicit status."),
    100 * config$effect_size$ci_level
  )
}

build_effect_concordance <- function(effect_sizes, taxonomy, config) {
  annotations <- dplyr::select(taxonomy, "Protein", "BiomarkerClass", "DirectionClass")
  table <- effect_sizes |>
    dplyr::select("Protein", "PD_HedgesG", "ALS_HedgesG", "EffectStatus_PD", "EffectStatus_ALS") |>
    dplyr::left_join(annotations, by = "Protein") |>
    dplyr::mutate(
      Recommended = !is.na(.data$BiomarkerClass),
      BothEffectsEstimable = is.finite(.data$PD_HedgesG) & is.finite(.data$ALS_HedgesG),
      CombinedAbsoluteEffect = abs(.data$PD_HedgesG) + abs(.data$ALS_HedgesG),
      LabelPriority = dplyr::case_when(
        .data$BiomarkerClass == "Shared_differential" & grepl("^Discordant_", .data$DirectionClass) ~ 1L,
        .data$BiomarkerClass == "PD_distinct" ~ 2L,
        .data$BiomarkerClass == "ALS_distinct" ~ 3L,
        TRUE ~ 4L
      ),
      DisplayLabel = FALSE
    )
  valid <- dplyr::filter(table, .data$BothEffectsEstimable)
  if (nrow(valid) >= 3) {
    test <- suppressWarnings(stats::cor.test(valid$PD_HedgesG, valid$ALS_HedgesG,
                                             method = config$effect_concordance$correlation_method,
                                             exact = FALSE))
    statistics <- data.frame(
      Method = config$effect_concordance$correlation_method,
      SpearmanRho = unname(test$estimate), PValue = test$p.value, NProteins = nrow(valid)
    )
  } else {
    statistics <- data.frame(Method = config$effect_concordance$correlation_method,
                             SpearmanRho = NA_real_, PValue = NA_real_, NProteins = nrow(valid))
  }
  candidates <- table |>
    dplyr::filter(.data$Recommended, .data$BothEffectsEstimable) |>
    dplyr::arrange(.data$LabelPriority, dplyr::desc(.data$CombinedAbsoluteEffect), .data$Protein) |>
    dplyr::slice_head(n = config$effect_concordance$max_labels)
  table$DisplayLabel[match(candidates$Protein, table$Protein)] <- TRUE
  list(table = table, statistics = statistics)
}

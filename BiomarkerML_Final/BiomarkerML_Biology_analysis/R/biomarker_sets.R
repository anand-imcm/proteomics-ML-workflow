build_biomarker_sets <- function(recommended_tables, config) {
  included <- lapply(recommended_tables, included_biomarkers, config = config)
  names(included) <- names(recommended_tables)
  all_proteins <- sort(unique(unlist(lapply(included, `[[`, "Protein"))))
  membership <- data.frame(Protein = all_proteins, stringsAsFactors = FALSE)
  for (nm in names(included)) membership[[nm]] <- membership$Protein %in% included[[nm]]$Protein
  membership <- dplyr::mutate(membership,
    AnalysisSupport = rowSums(dplyr::across(dplyr::all_of(names(included)))),
    # DEPRECATED compatibility output only. Current analyses must use
    # BiomarkerClass, DirectionClass, and raw PD/ALS/CrossDisease booleans.
    LegacyMembershipClass = dplyr::case_when(
      PD & ALS & CrossDisease ~ "PD_ALS_shared_CrossDisease_supported",
      PD & ALS ~ "PD_ALS_shared",
      PD & CrossDisease ~ "PD_specific_CrossDisease_supported",
      ALS & CrossDisease ~ "ALS_specific_CrossDisease_supported",
      PD ~ "PD_specific",
      ALS ~ "ALS_specific",
      CrossDisease ~ "CrossDisease_only",
      TRUE ~ "unclassified"))
  list(included = included, membership = membership)
}

classify_biomarker_architecture <- function(pd, als, cross_disease) {
  dplyr::case_when(
    pd & als & !cross_disease ~ "Shared",
    pd & als & cross_disease ~ "Shared_differential",
    pd & !als & !cross_disease ~ "PD_selective",
    !pd & als & !cross_disease ~ "ALS_selective",
    pd & !als & cross_disease ~ "PD_distinct",
    !pd & als & cross_disease ~ "ALS_distinct",
    !pd & !als & cross_disease ~ "Differential_only",
    TRUE ~ "Unclassified"
  )
}

classify_effect_direction <- function(pd_g, als_g) {
  dplyr::case_when(
    !is.finite(pd_g) | !is.finite(als_g) | pd_g == 0 | als_g == 0 ~ "Direction_uncertain",
    pd_g > 0 & als_g > 0 ~ "Concordant_up",
    pd_g < 0 & als_g < 0 ~ "Concordant_down",
    pd_g > 0 & als_g < 0 ~ "Discordant_PD_up_ALS_down",
    pd_g < 0 & als_g > 0 ~ "Discordant_PD_down_ALS_up",
    TRUE ~ "Direction_uncertain"
  )
}

build_revised_taxonomy <- function(membership, effect_sizes, included) {
  scores <- dplyr::bind_rows(lapply(names(included), function(nm) {
    included[[nm]] |>
      dplyr::transmute(Protein = .data$Protein, Analysis = nm,
                       RecommendedBiomarkerScore = .data$RecommendedBiomarkerScore)
  })) |>
    tidyr::pivot_wider(names_from = "Analysis", values_from = "RecommendedBiomarkerScore",
                       names_glue = "{Analysis}_RecommendedBiomarkerScore")
  membership |>
    dplyr::left_join(dplyr::select(effect_sizes, "Protein", "PD_HedgesG", "ALS_HedgesG"), by = "Protein") |>
    dplyr::mutate(
      BiomarkerClass = classify_biomarker_architecture(.data$PD, .data$ALS, .data$CrossDisease),
      DirectionClass = classify_effect_direction(.data$PD_HedgesG, .data$ALS_HedgesG)
    ) |>
    dplyr::left_join(scores, by = "Protein") |>
    dplyr::arrange(.data$Protein)
}

build_score_matrix <- function(included, membership, min_support = 2L) {
  keep <- membership$Protein[membership$AnalysisSupport >= min_support]
  out <- data.frame(Protein = keep, stringsAsFactors = FALSE)
  for (nm in names(included)) {
    lookup <- setNames(included[[nm]]$RecommendedBiomarkerScore, included[[nm]]$Protein)
    # Names absent from an analysis remain NA by design.
    out[[nm]] <- unname(lookup[out$Protein])
  }
  annotations <- intersect(c("Protein", "BiomarkerClass", "DirectionClass",
                             "PD_HedgesG", "ALS_HedgesG", "AnalysisSupport"), names(membership))
  dplyr::left_join(out, dplyr::select(membership, dplyr::all_of(annotations)), by = "Protein")
}

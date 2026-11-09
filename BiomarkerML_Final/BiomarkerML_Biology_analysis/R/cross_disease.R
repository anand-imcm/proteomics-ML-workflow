derive_pathway_support <- function(enrichment, membership, mapping, config) {
  if (!nrow(enrichment)) return(enrichment)
  significant <- dplyr::filter(enrichment, .data$AdjustedPValue < config$enrichment$fdr_threshold)
  # Map Entrez memberships back to original upstream protein labels; joins retain repeat aliases visibly.
  id_to_protein <- mapping |>
    dplyr::filter(.data$MappingStatus == "mapped") |>
    dplyr::select("OriginalProtein", "EnrichmentID")
  links <- significant |>
    tidyr::separate_rows("SupportingProteins", sep = ";") |>
    dplyr::rename(EnrichmentID = "SupportingProteins") |>
    dplyr::inner_join(id_to_protein, by = "EnrichmentID") |>
    dplyr::inner_join(dplyr::select(membership, dplyr::any_of(c("Protein", "BiomarkerClass", "DirectionClass", "PD", "ALS", "CrossDisease"))),
                      by = c("OriginalProtein" = "Protein"))
  links
}

biomarker_class_levels <- function() c(
  "Shared", "PD_distinct", "ALS_distinct", "Shared_differential",
  "PD_selective", "ALS_selective", "Differential_only"
)

build_biomarker_class_enrichment_sets <- function(taxonomy, mapping, config) {
  classes <- biomarker_class_levels()
  eligibility <- lapply(classes, function(class) {
    proteins <- taxonomy$Protein[taxonomy$BiomarkerClass == class]
    mapped <- mapping |>
      dplyr::filter(.data$OriginalProtein %in% proteins, .data$MappingStatus == "mapped") |>
      dplyr::distinct(.data$OriginalProtein)
    data.frame(BiomarkerClass = class, ProteinCount = length(unique(proteins)), MappedProteinCount = nrow(mapped),
      MinimumRequired = config$enrichment$min_gene_set_size,
      EligibleForORA = nrow(mapped) >= config$enrichment$min_gene_set_size,
      Status = if (nrow(mapped) >= config$enrichment$min_gene_set_size) "eligible" else "not_run: mapped_set_below_minimum",
      stringsAsFactors = FALSE)
  }) |>
    dplyr::bind_rows()
  eligible <- eligibility$BiomarkerClass[eligibility$EligibleForORA]
  sets <- setNames(lapply(eligible, function(class) {
    data.frame(Protein = taxonomy$Protein[taxonomy$BiomarkerClass == class], stringsAsFactors = FALSE)
  }), eligible)
  list(sets = sets, eligibility = eligibility)
}

derive_disease_pathway_links <- function(enrichment, membership, mapping, config) {
  if (!nrow(enrichment)) return(data.frame())
  significant <- enrichment |>
    dplyr::filter(.data$AdjustedPValue < config$enrichment$fdr_threshold, .data$InputSet %in% c("PD", "ALS"))
  id_to_protein <- mapping |>
    dplyr::filter(.data$MappingStatus == "mapped") |>
    dplyr::select("OriginalProtein", "EnrichmentID")
  significant |>
    tidyr::separate_rows("SupportingProteins", sep = ";") |>
    dplyr::rename(EnrichmentID = "SupportingProteins") |>
    dplyr::inner_join(id_to_protein, by = "EnrichmentID") |>
    dplyr::inner_join(
      dplyr::select(membership, "Protein", "PD", "ALS", "CrossDisease", "BiomarkerClass", "DirectionClass"),
      by = c("OriginalProtein" = "Protein")
    ) |>
    dplyr::filter((.data$InputSet == "PD" & .data$PD) | (.data$InputSet == "ALS" & .data$ALS)) |>
    dplyr::distinct(.data$Database, .data$TermID, .data$TermName, .data$InputSet,
                    .data$AdjustedPValue, .data$OriginalProtein, .data$BiomarkerClass,
                    .data$DirectionClass, .data$PD, .data$ALS, .data$CrossDisease)
}

calculate_pathway_convergence_metrics <- function(disease_links, config) {
  empty <- data.frame(Database=character(), TermID=character(), TermName=character(),
    PDAdjustedPValue=numeric(), ALSAdjustedPValue=numeric(), SharedPathwayFDR=numeric(),
    PDProteinCount=integer(), ALSProteinCount=integer(), IntersectionCount=integer(), UnionCount=integer(),
    ProteinJaccard=numeric(), TotalProteinCount=integer(), MinimumSideProteinCount=integer(),
    PDProteins=character(), ALSProteins=character(), SharedProteins=character(), Displayed=logical())
  if (!nrow(disease_links)) return(empty)
  sides <- disease_links |>
    dplyr::group_by(.data$Database, .data$TermID, .data$TermName, .data$InputSet) |>
    dplyr::summarise(
      AdjustedPValue = min(.data$AdjustedPValue),
      Proteins = paste(sort(unique(.data$OriginalProtein)), collapse = ";"),
      ProteinCount = dplyr::n_distinct(.data$OriginalProtein), .groups = "drop"
    )
  pd <- sides |>
    dplyr::filter(.data$InputSet == "PD") |>
    dplyr::transmute(.data$Database, .data$TermID, PDTermName = .data$TermName,
                     PDAdjustedPValue = .data$AdjustedPValue, PDProteinCount = .data$ProteinCount,
                     PDProteins = .data$Proteins)
  als <- sides |>
    dplyr::filter(.data$InputSet == "ALS") |>
    dplyr::transmute(.data$Database, .data$TermID, ALSTermName = .data$TermName,
                     ALSAdjustedPValue = .data$AdjustedPValue, ALSProteinCount = .data$ProteinCount,
                     ALSProteins = .data$Proteins)
  dplyr::inner_join(pd, als, by = c("Database", "TermID")) |>
    dplyr::rowwise() |>
    dplyr::mutate(
      TermName = dplyr::coalesce(.data$PDTermName, .data$ALSTermName),
      IntersectionCount = length(intersect(strsplit(.data$PDProteins, ";", fixed = TRUE)[[1]],
                                           strsplit(.data$ALSProteins, ";", fixed = TRUE)[[1]])),
      UnionCount = length(union(strsplit(.data$PDProteins, ";", fixed = TRUE)[[1]],
                               strsplit(.data$ALSProteins, ";", fixed = TRUE)[[1]])),
      SharedProteins = paste(sort(intersect(strsplit(.data$PDProteins, ";", fixed = TRUE)[[1]],
                                            strsplit(.data$ALSProteins, ";", fixed = TRUE)[[1]])), collapse = ";"),
      ProteinJaccard = ifelse(.data$UnionCount > 0, .data$IntersectionCount / .data$UnionCount, NA_real_),
      SharedPathwayFDR = max(.data$PDAdjustedPValue, .data$ALSAdjustedPValue),
      TotalProteinCount = .data$PDProteinCount + .data$ALSProteinCount,
      MinimumSideProteinCount = min(.data$PDProteinCount, .data$ALSProteinCount)
    ) |>
    dplyr::ungroup() |>
    dplyr::arrange(.data$SharedPathwayFDR, dplyr::desc(.data$TotalProteinCount),
                   dplyr::desc(.data$MinimumSideProteinCount), .data$TermName, .data$TermID) |>
    dplyr::mutate(Displayed = dplyr::row_number() <= config$visualization$max_shared_pathways_YD) |>
    dplyr::select("Database", "TermID", "TermName", "PDAdjustedPValue", "ALSAdjustedPValue",
                  "SharedPathwayFDR", "PDProteinCount", "ALSProteinCount", "IntersectionCount",
                  "UnionCount", "ProteinJaccard", "TotalProteinCount", "MinimumSideProteinCount",
                  "PDProteins", "ALSProteins", "SharedProteins", "Displayed")
}

build_convergence_edges <- function(pathway_links, pathway_selection) {
  if (!nrow(pathway_links) || !nrow(pathway_selection)) return(data.frame())
  displayed <- dplyr::filter(pathway_selection, .data$Displayed)
  edges <- pathway_links |>
    dplyr::inner_join(dplyr::select(displayed, "Database", "TermID"), by = c("Database", "TermID"))
  required <- c("BiomarkerClass", "DirectionClass", "InputSet")
  if (!all(required %in% names(edges)) || !all(edges$InputSet %in% c("PD", "ALS")))
    stopf("Y-D requires current disease-axis links with BiomarkerClass, DirectionClass, and PD/ALS InputSet")
  edges |>
    dplyr::group_by(.data$OriginalProtein, .data$TermName, .data$BiomarkerClass, .data$DirectionClass) |>
    dplyr::summarise(is_pd_supported = any(.data$InputSet == "PD"),
                     is_als_supported = any(.data$InputSet == "ALS"), .groups = "drop") |>
    dplyr::transmute(Protein = .data$OriginalProtein, Pathway = .data$TermName,
                     BiomarkerClass = .data$BiomarkerClass, DirectionClass = .data$DirectionClass,
                     is_pd_supported = .data$is_pd_supported, is_als_supported = .data$is_als_supported,
                     Source = .data$OriginalProtein, Target = .data$TermName)
}

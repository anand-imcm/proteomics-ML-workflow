require_bioc_mapping <- function() {
  if (!requireNamespace("AnnotationDbi", quietly = TRUE) || !requireNamespace("org.Hs.eg.db", quietly = TRUE))
    stopf("Protein mapping requires AnnotationDbi and org.Hs.eg.db. Install Bioconductor packages before running the analysis.")
}

map_proteins <- function(proteins) {
  require_bioc_mapping()
  proteins <- unique(trimws(as.character(proteins)))
  proteins <- proteins[!is.na(proteins) & nzchar(proteins)]
  keys <- AnnotationDbi::keys(org.Hs.eg.db::org.Hs.eg.db, keytype = "SYMBOL")
  is_symbol <- proteins %in% keys
  # SYMBOL is tried first; ALIAS covers common legacy symbols. No unmapped entry is discarded.
  symbol_map <- AnnotationDbi::select(org.Hs.eg.db::org.Hs.eg.db, keys = proteins[is_symbol],
                                      columns = c("SYMBOL", "ENTREZID"), keytype = "SYMBOL")
  if (nrow(symbol_map)) symbol_map$OriginalProtein <- as.character(symbol_map$SYMBOL)
  alias_candidates <- proteins[!is_symbol]
  alias_keys <- AnnotationDbi::keys(org.Hs.eg.db::org.Hs.eg.db, keytype = "ALIAS")
  alias_candidates <- intersect(alias_candidates, alias_keys)
  aliases <- if (length(alias_candidates)) AnnotationDbi::select(org.Hs.eg.db::org.Hs.eg.db, keys = alias_candidates,
    columns = c("SYMBOL", "ENTREZID"), keytype = "ALIAS") else data.frame(ALIAS=character(), SYMBOL=character(), ENTREZID=character())
  if (nrow(aliases)) aliases$OriginalProtein <- as.character(aliases$ALIAS)
  map_rows <- dplyr::bind_rows(
    dplyr::select(symbol_map, dplyr::any_of(c("OriginalProtein", "SYMBOL", "ENTREZID"))),
    dplyr::select(aliases, dplyr::any_of(c("OriginalProtein", "SYMBOL", "ENTREZID")))
  )
  if (nrow(map_rows)) {
    symbol_map <- map_rows |>
      dplyr::filter(!is.na(.data$ENTREZID)) |>
      dplyr::arrange(.data$OriginalProtein, .data$ENTREZID) |>
      dplyr::group_by(.data$OriginalProtein) |>
      dplyr::slice(1) |>
      dplyr::ungroup() |>
      dplyr::transmute(OriginalProtein = .data$OriginalProtein, GeneSymbol = .data$SYMBOL,
                       EnrichmentID = as.character(.data$ENTREZID))
  } else symbol_map <- data.frame(OriginalProtein = character(), GeneSymbol = character(), EnrichmentID = character())
  out <- dplyr::left_join(data.frame(OriginalProtein = proteins), symbol_map, by = "OriginalProtein")
  out$MappingStatus <- ifelse(is.na(out$EnrichmentID), "unmapped", "mapped")
  out
}

mapping_summary <- function(mapping) data.frame(
  Metric = c("input_proteins", "successfully_mapped", "unmapped", "mapping_percentage"),
  Value = c(nrow(mapping), sum(mapping$MappingStatus == "mapped"), sum(mapping$MappingStatus == "unmapped"),
            round(100 * mean(mapping$MappingStatus == "mapped"), 2))
)

map_analysis_sets <- function(included, expression) {
  ids <- unique(c(unlist(lapply(included, `[[`, "Protein")), unique(expression$Protein)))
  map_proteins(ids)
}

build_background_universe <- function(expression, mapping) {
  universes <- derive_protein_universes(expression, mapping)
  ids <- universes$EnrichmentUniverse$EnrichmentID
  ids <- unique(stats::na.omit(ids))
  if (length(ids) < 3) stopf("Mapped measurable background has fewer than 3 proteins; enrichment cannot be validly run.")
  ids
}

derive_protein_universes <- function(expression, mapping) {
  measured <- sort(unique(as.character(expression$Protein[is.finite(expression$Abundance)])))
  measured <- measured[!is.na(measured) & nzchar(measured)]
  mapping_view <- dplyr::select(mapping, dplyr::any_of(c("OriginalProtein", "GeneSymbol", "EnrichmentID", "MappingStatus")))
  if (!"GeneSymbol" %in% names(mapping_view)) mapping_view$GeneSymbol <- NA_character_
  measured_table <- data.frame(Protein = measured, stringsAsFactors = FALSE) |>
    dplyr::left_join(
      dplyr::rename(mapping_view, Protein = "OriginalProtein"),
      by = "Protein"
    ) |>
    dplyr::mutate(MappingStatus = dplyr::coalesce(.data$MappingStatus, "unmapped"))
  enrichment <- measured_table |>
    dplyr::filter(.data$MappingStatus == "mapped", !is.na(.data$EnrichmentID), nzchar(.data$EnrichmentID)) |>
    dplyr::distinct(.data$Protein, .data$GeneSymbol, .data$EnrichmentID, .data$MappingStatus)
  list(MeasuredCommonUniverse = measured_table, EnrichmentUniverse = enrichment)
}

universe_summary <- function(universes) data.frame(
  Metric = c("MeasuredCommonUniverse", "EnrichmentUniverse", "MappedMeasuredProteins", "UnmappedMeasuredProteins"),
  Value = c(
    nrow(universes$MeasuredCommonUniverse),
    nrow(universes$EnrichmentUniverse),
    sum(universes$MeasuredCommonUniverse$MappingStatus == "mapped"),
    sum(universes$MeasuredCommonUniverse$MappingStatus != "mapped")
  )
)

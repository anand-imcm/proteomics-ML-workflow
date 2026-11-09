require_enrichment_packages <- function() {
  needed <- c("clusterProfiler", "ReactomePA", "org.Hs.eg.db")
  absent <- needed[!vapply(needed, requireNamespace, logical(1), quietly = TRUE)]
  if (length(absent)) stopf("Enrichment requires unavailable package(s): %s", paste(absent, collapse = ", "))
}

tidy_enrichment <- function(x, database, input_set, background_n) {
  empty <- data.frame(TermID=character(), TermName=character(), Database=character(), InputSet=character(), GeneCount=integer(), BackgroundCount=integer(), GeneRatio=numeric(), BackgroundRatio=numeric(), EnrichmentRatio=numeric(), PValue=numeric(), AdjustedPValue=numeric(), SupportingProteins=character())
  if (is.null(x) || !nrow(as.data.frame(x))) return(empty)
  d <- as.data.frame(x)
  ratio <- function(v) vapply(strsplit(as.character(v), "/", fixed = TRUE), function(z) as.numeric(z[1]) / as.numeric(z[2]), numeric(1))
  data.frame(TermID = d$ID, TermName = d$Description, Database = database, InputSet = input_set,
    GeneCount = d$Count, BackgroundCount = round(ratio(d$BgRatio) * background_n), GeneRatio = ratio(d$GeneRatio),
    BackgroundRatio = ratio(d$BgRatio), EnrichmentRatio = ratio(d$GeneRatio) / ratio(d$BgRatio),
    PValue = d$pvalue, AdjustedPValue = d$p.adjust,
    SupportingProteins = gsub("/", ";", d$geneID, fixed = TRUE), stringsAsFactors = FALSE)
}

run_ora <- function(protein_ids, universe_ids, input_set, config) {
  if (isTRUE(config$project$synthetic_fixture_mode)) {
    ids <- sort(intersect(unique(as.character(protein_ids)), as.character(universe_ids)))
    if (length(ids) < config$enrichment$min_gene_set_size)
      return(tidy_enrichment(NULL, "none", input_set, length(universe_ids)))
    databases <- config$enrichment$databases
    if (isTRUE(config$enrichment$include_KEGG)) databases <- unique(c(databases, "KEGG"))
    # This deterministic, explicitly labelled branch exists only for the isolated
    # end-to-end fixture test. Production configuration never enables it.
    return(dplyr::bind_rows(lapply(seq_along(databases), function(i) {
      data.frame(
        TermID = paste0("SYNTHETIC_FIXTURE_", databases[i]),
        TermName = paste("Synthetic fixture", gsub("_", " ", databases[i], fixed = TRUE), "process"),
        Database = databases[i], InputSet = input_set,
        GeneCount = length(ids), BackgroundCount = length(universe_ids),
        GeneRatio = 1, BackgroundRatio = length(ids) / length(universe_ids),
        EnrichmentRatio = length(universe_ids) / length(ids),
        PValue = .001 * i, AdjustedPValue = .01 * i,
        SupportingProteins = paste(ids, collapse = ";"), stringsAsFactors = FALSE
      )
    })))
  }
  require_enrichment_packages()
  ids <- intersect(unique(protein_ids), universe_ids)
  if (length(ids) < config$enrichment$min_gene_set_size) return(tidy_enrichment(NULL, "none", input_set, length(universe_ids)))
  common <- list(genes = ids, universe = universe_ids, pAdjustMethod = config$enrichment$adjustment_method,
                 pvalueCutoff = 1, qvalueCutoff = 1, minGSSize = config$enrichment$min_gene_set_size, maxGSSize = config$enrichment$max_gene_set_size)
  results <- list()
  if ("GO_BP" %in% config$enrichment$databases) {
    go <- do.call(clusterProfiler::enrichGO, c(common, list(OrgDb = org.Hs.eg.db::org.Hs.eg.db, ont = "BP", keyType = "ENTREZID", readable = FALSE)))
    results[["GO_BP"]] <- tidy_enrichment(go, "GO_BP", input_set, length(universe_ids))
  }
  if ("Reactome" %in% config$enrichment$databases) {
    re <- do.call(ReactomePA::enrichPathway, c(common, list(organism = "human", readable = FALSE)))
    results[["Reactome"]] <- tidy_enrichment(re, "Reactome", input_set, length(universe_ids))
  }
  if (isTRUE(config$enrichment$include_KEGG)) {
    ke <- do.call(clusterProfiler::enrichKEGG, c(common, list(organism = "hsa")))
    results[["KEGG"]] <- tidy_enrichment(ke, "KEGG", input_set, length(universe_ids))
  }
  dplyr::bind_rows(results)
}

run_enrichment_sets <- function(sets, mapping, universe, config) {
  if (!length(sets)) return(tidy_enrichment(NULL, "none", "none", length(universe)))
  result <- lapply(names(sets), function(nm) {
    map <- dplyr::filter(mapping, .data$OriginalProtein %in% (sets[[nm]])$Protein, .data$MappingStatus == "mapped")
    run_ora(map$EnrichmentID, universe, nm, config)
  })
  names(result) <- names(sets)
  dplyr::bind_rows(result)
}

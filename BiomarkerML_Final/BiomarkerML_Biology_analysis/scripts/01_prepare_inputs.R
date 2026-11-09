source("R/io_validation.R"); source("R/biomarker_sets.R"); source("R/protein_mapping.R"); source("R/effect_sizes.R")
dir.create("results", showWarnings = FALSE, recursive = TRUE); dir.create("logs", showWarnings = FALSE, recursive = TRUE)
for (path in c("results/effect_sizes", "results/biomarker_sets", "results/cross_disease",
               "results/enrichment", "results/pathway_mapping", "results/correlations",
               "figures/individual_panels", "figures/assembled", "figures/supplementary"))
  dir.create(path, showWarnings = FALSE, recursive = TRUE)
config <- read_config(); set.seed(config$project$seed)
paths <- input_paths()
tables <- list(PD = read_recommended_biomarkers(paths$PD, "PD"), ALS = read_recommended_biomarkers(paths$ALS, "ALS"), CrossDisease = read_recommended_biomarkers(paths$CrossDisease, "CrossDisease"))
expression <- read_expression_matrix(paths$expression, config)
sets <- build_biomarker_sets(tables, config)
mapping <- map_analysis_sets(sets$included, expression)
universes <- derive_protein_universes(expression, mapping)
universe <- build_background_universe(expression, mapping)
effect_sizes <- calculate_all_protein_effect_sizes(expression, universes$MeasuredCommonUniverse$Protein, config)
taxonomy <- build_revised_taxonomy(sets$membership, effect_sizes, sets$included)
recommended_effects <- recommended_effect_sizes(effect_sizes, sets$included) |>
  dplyr::left_join(dplyr::select(taxonomy, "Protein", "BiomarkerClass", "DirectionClass"), by = "Protein")
score_matrix <- build_score_matrix(sets$included, taxonomy, config$visualization$heatmap_min_analysis_support)
readr::write_csv(mapping, "results/protein_identifier_mapping.csv")
readr::write_csv(mapping_summary(mapping), "results/protein_identifier_mapping_summary.csv")
readr::write_csv(universes$MeasuredCommonUniverse, "results/measured_common_universe.csv")
readr::write_csv(universes$EnrichmentUniverse, "results/enrichment_universe.csv")
readr::write_csv(universe_summary(universes), "results/protein_universe_summary.csv")
readr::write_csv(effect_sizes, "results/effect_sizes/all_protein_effect_sizes.csv")
readr::write_csv(recommended_effects, "results/effect_sizes/recommended_biomarker_effect_sizes.csv")
writeLines(effect_size_method_note(config), "results/effect_sizes/method_note.txt")
readr::write_csv(taxonomy, "results/biomarker_sets/biomarker_membership_classification.csv")
readr::write_csv(taxonomy, "results/biomarker_sets/revised_biomarker_taxonomy.csv")
readr::write_csv(score_matrix, "results/cross_disease/recommendation_score_matrix.csv")
readr::write_csv(data.frame(EnrichmentID = universe), "results/background_universe.csv")
stat_meta <- data.frame(Analysis = c("PD", "ALS", "CrossDisease"), Test = "ORA (hypergeometric)",
  MeasuredCommonUniverseSize = nrow(universes$MeasuredCommonUniverse), BackgroundUniverseSize = length(universe),
  FDRMethod = config$enrichment$adjustment_method, FDRThreshold = config$enrichment$fdr_threshold, stringsAsFactors = FALSE)
readr::write_csv(stat_meta, "results/statistical_metadata.csv")
saveRDS(list(config=config, tables=tables, included=sets$included, membership=taxonomy, raw_membership=sets$membership,
  expression=expression, mapping=mapping, universes=universes, universe=universe, effect_sizes=effect_sizes,
  taxonomy=taxonomy, score_matrix=score_matrix), "results/prepared_data.rds")
manifest <- list(analysis_date = as.character(Sys.Date()), input_paths = paths, input_md5 = as.list(tools::md5sum(unlist(paths))),
  biomarker_inclusion_rule = "Recommended == TRUE", cross_disease_definition = config$cross_disease$definition,
  score_policy = "Read directly from upstream; missing score stays NA",
  measured_common_universe_size = nrow(universes$MeasuredCommonUniverse), enrichment_universe_size = length(universe),
  background_universe = config$enrichment$background_strategy, effect_size = config$effect_size,
  fdr = list(method=config$enrichment$adjustment_method, threshold=config$enrichment$fdr_threshold),
  redundancy = list(jaccard_threshold=config$enrichment$redundancy_jaccard_threshold), correlation = config$correlation,
  visualization = config$visualization, figure = config$figure)
yaml::write_yaml(manifest, "analysis_manifest.yaml")
writeLines(capture.output(sessionInfo()), "sessionInfo.txt")
ip <- installed.packages(); readr::write_csv(data.frame(Package=ip[,"Package"], Version=ip[,"Version"]), "package_versions.csv")
message("Prepared immutable upstream inputs, distinct protein universes, effect sizes, and revised taxonomy.")

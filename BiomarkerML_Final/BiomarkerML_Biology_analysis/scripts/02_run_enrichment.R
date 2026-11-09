source("R/io_validation.R"); source("R/enrichment.R"); source("R/redundancy_reduction.R")
if (!file.exists("results/prepared_data.rds")) source("scripts/01_prepare_inputs.R")
d <- readRDS("results/prepared_data.rds"); config <- d$config
full <- run_enrichment_sets(d$included[c("PD", "ALS")], d$mapping, d$universe, config)
reduced <- reduce_pathway_redundancy(full, config$enrichment$redundancy_jaccard_threshold)
readr::write_csv(full, "results/enrichment/PD_ALS_full_enrichment.csv")
readr::write_csv(reduced, "results/enrichment/PD_ALS_nonredundant_enrichment.csv")
saveRDS(list(full=full, reduced=reduced), "results/enrichment/pd_als_enrichment.rds")
message("Saved complete and deterministic non-redundant PD/ALS enrichment results.")

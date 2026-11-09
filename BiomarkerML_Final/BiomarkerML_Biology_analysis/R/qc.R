figure_qc <- function(config, expected = c("Figure_XA", "Figure_XB", "Figure_XC", "Figure_YA", "Figure_YB", "Figure_YC", "Figure_YD")) {
  records <- list()
  paths <- c(file.path("figures/individual_panels", expected), file.path("figures/assembled", c("Figure_X", "Figure_Y")))
  for (stem in paths) for (ext in c("png", "tiff", "pdf")) {
    path <- paste0(stem, ".", ext); records[[length(records)+1]] <- data.frame(Check = paste("exists", basename(path)), Pass = file.exists(path), Detail = path)
    if (file.exists(path) && ext != "pdf" && requireNamespace("magick", quietly = TRUE)) {
      info <- magick::image_info(magick::image_read(path)); records[[length(records)+1]] <- data.frame(Check = paste("raster_dpi", basename(path)), Pass = !is.na(info$density) && grepl("600", info$density), Detail = as.character(info$density))
    }
  }
  records[[length(records)+1]] <- data.frame(Check = "assembled_width_mm", Pass = config$figure$width_mm <= 180, Detail = as.character(config$figure$width_mm))
  records[[length(records)+1]] <- data.frame(Check = "configured_raster_dpi_at_least_300", Pass = config$figure$dpi >= 300, Detail = as.character(config$figure$dpi))
  records[[length(records)+1]] <- data.frame(Check = "configured_default_raster_dpi_600", Pass = identical(as.numeric(config$figure$dpi), 600), Detail = as.character(config$figure$dpi))
  yb_path <- "results/cross_disease/recommendation_score_matrix.csv"
  if (file.exists(yb_path)) {
    yb <- readr::read_csv(yb_path, show_col_types = FALSE)
    score_cols <- intersect(c("PD", "ALS", "CrossDisease"), names(yb))
    records[[length(records)+1]] <- data.frame(Check = "YB_not_all_NA", Pass = length(score_cols) > 0 && any(is.finite(as.matrix(yb[score_cols]))), Detail = yb_path)
  }
  nr_paths <- c("results/enrichment/PD_ALS_nonredundant_enrichment.csv", "results/cross_disease/nonredundant_enrichment.csv")
  for (p in nr_paths[file.exists(nr_paths)]) {
    nr <- readr::read_csv(p, show_col_types = FALSE)
    finite <- !"AdjustedPValue" %in% names(nr) || all(is.finite(nr$AdjustedPValue) | is.na(nr$AdjustedPValue))
    nodup <- !all(c("InputSet", "Database", "TermName") %in% names(nr)) || !anyDuplicated(nr[c("InputSet", "Database", "TermName")])
    records[[length(records)+1]] <- data.frame(Check = paste("finite_adjusted_p", basename(p)), Pass = finite, Detail = p)
    records[[length(records)+1]] <- data.frame(Check = paste("unique_nonredundant_labels", basename(p)), Pass = nodup, Detail = p)
  }
  source_tables <- c("results/pathway_mapping/Figure_XA_source_table.csv", "results/pathway_mapping/Figure_XB_source_table.csv", "results/correlations/Figure_XC_source_table.csv", "results/biomarker_sets/Figure_YA_source_table.csv", "results/cross_disease/Figure_YB_source_table.csv", "results/cross_disease/Figure_YC_source_table.csv", "results/pathway_mapping/Figure_YD_source_table.csv", "results/pathway_mapping/Figure_YD_pathway_metrics_source_table.csv")
  records[[length(records)+1]] <- data.frame(Check = "all_panel_source_tables_exist", Pass = all(file.exists(source_tables)), Detail = paste(source_tables[!file.exists(source_tables)], collapse = ";"))
  required_upgrade_outputs <- c(
    "results/measured_common_universe.csv", "results/enrichment_universe.csv",
    "results/protein_universe_summary.csv",
    "results/effect_sizes/all_protein_effect_sizes.csv",
    "results/effect_sizes/recommended_biomarker_effect_sizes.csv",
    "results/biomarker_sets/revised_biomarker_taxonomy.csv",
    "results/cross_disease/pathway_convergence_metrics.csv",
    "results/pathway_mapping/YD_convergence_edges.csv"
  )
  records[[length(records)+1]] <- data.frame(
    Check = "all_upgrade_outputs_exist", Pass = all(file.exists(required_upgrade_outputs)),
    Detail = paste(required_upgrade_outputs[!file.exists(required_upgrade_outputs)], collapse = ";"))
  supplementary_outputs <- unlist(lapply(c("PD", "ALS"), function(side)
    file.path("figures", "supplementary", paste0("Supp_Figure_Correlation_", side, ".", c("png", "tiff", "pdf")))))
  records[[length(records)+1]] <- data.frame(
    Check = "supplementary_correlations_exist", Pass = all(file.exists(supplementary_outputs)),
    Detail = paste(supplementary_outputs[!file.exists(supplementary_outputs)], collapse = ";"))
  effect_path <- "results/effect_sizes/all_protein_effect_sizes.csv"
  if (file.exists(effect_path)) {
    effects <- readr::read_csv(effect_path, show_col_types = FALSE)
    analytical_columns <- grep("HedgesG|CI_Lower|CI_Upper", names(effects), value = TRUE)
    finite_or_na <- all(vapply(effects[analytical_columns], function(x) all(is.finite(x) | is.na(x)), logical(1)))
    records[[length(records)+1]] <- data.frame(Check = "effect_outputs_have_no_Inf_or_NaN", Pass = finite_or_na, Detail = effect_path)
  }
  records[[length(records)+1]] <- data.frame(Check = "mapping_reports_unmapped_proteins", Pass = file.exists("results/protein_identifier_mapping.csv"), Detail = "results/protein_identifier_mapping.csv")
  source_files <- list.files("R", full.names=TRUE)
  source_files <- source_files[basename(source_files) != "qc.R"]
  source_text <- unlist(lapply(source_files, readLines), use.names = FALSE)
  prohibited_call <- grepl("STRINGdb|STRING_db|\\bget_interaction|\\bppi\\s*\\(", source_text, ignore.case = TRUE)
  records[[length(records)+1]] <- data.frame(Check = "no_STRING_or_PPI_calls", Pass = !any(prohibited_call), Detail = "Source-code scan")
  reconstruction <- grepl("RecommendedBiomarkerScore\\s*(<-|=).*?(MeanSHAP|ConsensusScore|StabilityScore)", source_text)
  records[[length(records)+1]] <- data.frame(Check = "no_score_reconstruction", Pass = !any(reconstruction), Detail = "Source-code scan")
  out <- dplyr::bind_rows(records)
  readr::write_csv(out, "logs/figure_qc_report.csv")
  writeLines(c(sprintf("QC %s", if (all(out$Pass)) "PASS" else "FAIL"), apply(out, 1, function(z) sprintf("%s: %s (%s)", z[["Check"]], z[["Pass"]], z[["Detail"]]))), "logs/figure_qc_report.txt")
  out
}

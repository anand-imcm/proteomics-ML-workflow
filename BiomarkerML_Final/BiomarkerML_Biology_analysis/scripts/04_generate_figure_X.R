source("R/io_validation.R"); source("R/correlation.R"); source("R/effect_sizes.R"); source("R/plot_theme.R"); source("R/plot_figure_X.R"); source("R/figure_export.R")
if (!file.exists("results/prepared_data.rds")) source("scripts/01_prepare_inputs.R")
if (!file.exists("results/enrichment/pd_als_enrichment.rds")) source("scripts/02_run_enrichment.R")
d <- readRDS("results/prepared_data.rds"); e <- readRDS("results/enrichment/pd_als_enrichment.rds"); config <- d$config
selected <- select_terms(e$reduced, config$visualization$max_pathways_XB, config$enrichment$fdr_threshold)
xa_selected <- select_terms(e$reduced, config$visualization$max_pathways_XA, config$enrichment$fdr_threshold)
complete_links <- pathway_protein_links(selected, d$mapping, d$included)
display_selection <- select_display_biomarkers_for_pathway_map(complete_links, config$visualization$max_biomarkers_XB)
links <- if (nrow(display_selection)) complete_links |>
  dplyr::semi_join(dplyr::filter(display_selection, .data$Displayed), by = "CandidateKey") else complete_links
correlations <- calculate_disease_correlations(d$expression, d$included, config)
concordance <- build_effect_concordance(d$effect_sizes, d$taxonomy, config)
readr::write_csv(xa_selected, "results/pathway_mapping/Figure_XA_source_table.csv")
readr::write_csv(complete_links, "results/pathway_mapping/Figure_XB_complete_mapping.csv")
readr::write_csv(display_selection |> dplyr::select(-dplyr::any_of("CandidateKey")), "results/pathway_mapping/Figure_XB_display_selection.csv")
readr::write_csv(links, "results/pathway_mapping/Figure_XB_source_table.csv")
readr::write_csv(concordance$table, "results/effect_sizes/Figure_XC_effect_concordance_source_table.csv")
readr::write_csv(concordance$table, "results/correlations/Figure_XC_source_table.csv")
readr::write_csv(concordance$statistics, "results/effect_sizes/Figure_XC_effect_concordance_statistics.csv")
correlation_sources <- dplyr::bind_rows(Map(correlation_source_table, correlations, names(correlations)))
readr::write_csv(correlation_sources, "results/correlations/Supp_Figure_Correlation_source_table.csv")
readr::write_csv(data.frame(Analysis=names(correlations), GroupLabel=vapply(correlations, `[[`, character(1), "group_label"), Samples=vapply(correlations, `[[`, integer(1), "n"), Biomarkers=vapply(correlations, `[[`, integer(1), "biomarkers"), MeanMissingFraction=vapply(correlations, function(x) mean(x$missing_fraction), numeric(1)), MissingProteins=vapply(correlations, function(x) paste(x$missing, collapse=";"), character(1)), Method=config$correlation$method, ObservationUse=config$correlation$use), "results/correlations/correlation_metadata.csv")
stat_path <- "results/statistical_metadata.csv"
stat_meta <- if (file.exists(stat_path)) readr::read_csv(stat_path, show_col_types = FALSE) else data.frame()
cor_meta <- data.frame(Analysis = paste0(names(correlations), "_correlation"), Test = "Spearman correlation",
  Samples = vapply(correlations, `[[`, integer(1), "n"), Biomarkers = vapply(correlations, `[[`, integer(1), "biomarkers"),
  ObservationUse = config$correlation$use, FDRMethod = "BH", stringsAsFactors = FALSE)
readr::write_csv(dplyr::bind_rows(stat_meta, cor_meta), stat_path)
xa <- plot_xa(e$reduced, config); xb <- plot_xb(links, config)
xc <- plot_xc_effect_concordance(concordance$table, concordance$statistics, config)
save_panel_figure(xa, "Figure_XA", 82, 84, config); save_panel_figure(xb, "Figure_XB", 98, 84, config)
save_panel_figure(xc, "Figure_XC", 180, 72, config)
if (isTRUE(config$supplementary$export_correlation_heatmaps)) {
  dir.create("figures/supplementary", recursive = TRUE, showWarnings = FALSE)
  for (nm in intersect(c("PD", "ALS"), names(correlations))) {
    supp <- plot_xc(setNames(list(correlations[[nm]]), nm), config)
    supp_grob <- grid::grid.grabExpr(ComplexHeatmap::draw(supp, merge_legend = TRUE,
      heatmap_legend_side = "bottom"), wrap = TRUE, wrap.grobs = TRUE)
    save_grid_formats(function() grid::grid.draw(supp_grob),
      file.path("figures", "supplementary", paste0("Supp_Figure_Correlation_", nm)), 88, 78, config)
    readr::write_csv(correlation_source_table(correlations[[nm]], nm),
      file.path("results", "correlations", paste0("Supp_Figure_Correlation_", nm, "_source_table.csv")))
  }
}
if (!requireNamespace("patchwork", quietly=TRUE)) stopf("patchwork is required for assembled figures")
top_row <- (xa | xb) + patchwork::plot_layout(widths = c(.9, 1.1))
assembled <- top_row / xc + patchwork::plot_layout(heights = c(.53, .47))
save_assembled_figure(assembled, "Figure_X", config$figure$width_mm, 150, config)
write_figure_legends(concordance$statistics, config)
message("Generated Figure X panels and assembly.")

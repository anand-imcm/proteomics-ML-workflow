mm_to_in <- function(mm) mm / 25.4

validate_figure_dimensions <- function(width_mm) {
  if (width_mm > 180) stopf("Assembled figures may not exceed 180 mm; got %.1f mm", width_mm)
}

save_plot_formats <- function(plot, stem, width_mm, height_mm, config, assembled = FALSE) {
  if (assembled) validate_figure_dimensions(width_mm)
  dir.create(dirname(stem), recursive = TRUE, showWarnings = FALSE)
  fmts <- config$figure$output_formats
  for (fmt in fmts) {
    path <- paste0(stem, ".", fmt)
    if (fmt == "pdf") ggplot2::ggsave(path, plot, width = mm_to_in(width_mm), height = mm_to_in(height_mm), units = "in", device = grDevices::pdf, useDingbats = FALSE)
    else if (fmt == "tiff") ggplot2::ggsave(path, plot, width = mm_to_in(width_mm), height = mm_to_in(height_mm), units = "in", dpi = config$figure$dpi, device = "tiff", compression = "lzw")
    else ggplot2::ggsave(path, plot, width = mm_to_in(width_mm), height = mm_to_in(height_mm), units = "in", dpi = config$figure$dpi)
  }
}

save_panel_figure <- function(plot, filename_stem, width_mm = 85, height_mm = 75, config) save_plot_formats(plot, file.path("figures", "individual_panels", filename_stem), width_mm, height_mm, config)
save_assembled_figure <- function(plot, filename_stem, width_mm = 180, height_mm = 160, config) save_plot_formats(plot, file.path("figures", "assembled", filename_stem), width_mm, height_mm, config, assembled = TRUE)

capture_heatmap_grob <- function(heatmap, label, config, legend_side = "bottom") {
  body <- grid::grid.grabExpr(
    ComplexHeatmap::draw(
      heatmap, merge_legend = TRUE, heatmap_legend_side = legend_side,
      annotation_legend_side = legend_side
    ),
    wrap = TRUE, wrap.grobs = TRUE
  )
  label_grob <- grid::textGrob(
    label, x = grid::unit(0.8, "mm"), y = grid::unit(.5, "npc"),
    just = c("left", "centre"),
    gp = grid::gpar(
      fontsize = config$figure$panel_label_pt, fontface = "bold",
      fontfamily = config$figure$font_family, col = "#111111"
    )
  )
  gridExtra::arrangeGrob(
    label_grob, body, ncol = 1,
    heights = grid::unit.c(grid::unit(5, "mm"), grid::unit(1, "null"))
  )
}

save_grid_formats <- function(draw_fun, stem, width_mm, height_mm, config, assembled = FALSE) {
  if (assembled) validate_figure_dimensions(width_mm)
  dir.create(dirname(stem), recursive = TRUE, showWarnings = FALSE)
  for (fmt in config$figure$output_formats) {
    path <- paste0(stem, ".", fmt)
    if (fmt == "pdf") grDevices::pdf(path, width = mm_to_in(width_mm), height = mm_to_in(height_mm), useDingbats = FALSE)
    else if (fmt == "tiff" && requireNamespace("ragg", quietly = TRUE)) ragg::agg_tiff(path, width = mm_to_in(width_mm), height = mm_to_in(height_mm), units = "in", res = config$figure$dpi, compression = "lzw")
    else if (fmt == "tiff") grDevices::tiff(path, width = mm_to_in(width_mm), height = mm_to_in(height_mm), units = "in", res = config$figure$dpi, compression = "lzw")
    else if (requireNamespace("ragg", quietly = TRUE)) ragg::agg_png(path, width = mm_to_in(width_mm), height = mm_to_in(height_mm), units = "in", res = config$figure$dpi)
    else grDevices::png(path, width = mm_to_in(width_mm), height = mm_to_in(height_mm), units = "in", res = config$figure$dpi)
    draw_fun(); grDevices::dev.off()
  }
}

write_figure_legends <- function(concordance_statistics, config) {
  concordance_text <- sprintf("Spearman rho = %.2f; n = %d; two-sided P = %s",
    concordance_statistics$SpearmanRho, concordance_statistics$NProteins,
    format.pval(concordance_statistics$PValue, digits = 2, eps = 1e-4))
  writeLines(c(
    "Figure X | Disease-level biological characterization of BiomarkerML-derived panels.",
    "a, Comparative over-representation analysis of recommended PD and ALS biomarkers. Points show non-redundant GO Biological Process and Reactome terms; position is enrichment ratio, size is number of supporting biomarkers, and colour is −log10(BH-adjusted P value).",
    "b, Left-to-right annotation flows connecting recommended biomarkers to selected enriched pathways; ribbon and block colours distinguish PD and ALS input panels.",
    sprintf("c, Molecular effect concordance across all proteins in the MeasuredCommonUniverse with estimable PD-versus-control and ALS-versus-control Hedges' g values (%s). Grey points are background proteins; highlighted points are frozen Recommended biomarkers. The P value is reported for completeness and is not interpreted as effect importance.", concordance_text),
    "ORA used the mapped measurable-proteome background with Benjamini–Hochberg correction; pathways displayed as significant satisfy the configured FDR threshold."), "figures/assembled/Figure_X_legend_draft.txt")
  writeLines(c(
    "Figure Y | Cross-disease convergence and divergence of BiomarkerML-derived biomarker signals.",
    "a, Recommended-biomarker set intersections across PD, ALS and cross-disease analyses.",
    "b, Upstream RecommendedBiomarkerScore values (NA denotes absence from an analysis); compact annotations show mutually exclusive BiomarkerClass and independent DirectionClass.",
    "c, BiomarkerClass pathway-enrichment matrix. Point size is supporting biomarker count and colour is −log10(BH-adjusted P value); blank cells denote non-significant, absent or ineligible combinations.",
    "d, Annotation-membership architecture linking PD-axis, shared and ALS-axis Recommended proteins to pathways enriched on both disease sides. Edges denote annotation membership only; ProteinJaccard is reported continuously in the accompanying pathway-metrics source table.",
    "Disease colours are used consistently throughout; enrichment used the mapped measurable-proteome background and Benjamini–Hochberg correction."), "figures/assembled/Figure_Y_legend_draft.txt")
}

select_terms <- function(enrichment, n, fdr_threshold = 0.05) {
  enrichment |>
    dplyr::filter(.data$AdjustedPValue < fdr_threshold) |>
    dplyr::arrange(.data$AdjustedPValue, dplyr::desc(.data$GeneCount), .data$TermName) |>
    dplyr::group_by(.data$InputSet) |>
    dplyr::slice_head(n = n) |>
    dplyr::ungroup()
}

plot_xa <- function(nonredundant, config) {
  d <- select_terms(nonredundant, config$visualization$max_pathways_XA, config$enrichment$fdr_threshold)
  if (!nrow(d)) return(empty_panel("No significant PD or ALS pathways", config))
  d$InputSet <- factor(d$InputSet, levels = c("PD", "ALS"))
  d <- d |>
    dplyr::group_by(.data$InputSet) |>
    dplyr::arrange(.data$EnrichmentRatio, .by_group = TRUE) |>
    dplyr::mutate(TermKey = factor(paste(.data$InputSet, .data$TermName, sep = "||"),
                                  levels = paste(.data$InputSet, .data$TermName, sep = "||"))) |>
    dplyr::ungroup()
  p <- ggplot2::ggplot(d, ggplot2::aes(.data$EnrichmentRatio, .data$TermKey)) +
    ggplot2::geom_hline(ggplot2::aes(yintercept = as.numeric(.data$TermKey)), colour = COLORS$Grid, linewidth = 0.25) +
    ggplot2::geom_point(
      ggplot2::aes(size = .data$GeneCount, fill = safe_neglog10(.data$AdjustedPValue)),
      shape = 21, colour = "#344054", stroke = 0.25
    ) +
    ggplot2::facet_grid(.data$InputSet ~ ., scales = "free_y", space = "free_y", switch = "y") +
    scale_significance_fill(name = expression(-log[10](FDR))) +
    ggplot2::scale_y_discrete(labels = function(x) stringr::str_wrap(sub("^[^|]+\\|\\|", "", x), 30), expand = ggplot2::expansion(add = .55)) +
    ggplot2::scale_x_continuous(expand = ggplot2::expansion(mult = c(.02, .08))) +
    ggplot2::scale_size_area(max_size = 4.1, name = "Count") +
    ggplot2::labs(x = "Enrichment ratio", y = NULL) +
    ggplot2::guides(
      fill = ggplot2::guide_colourbar(title.position = "top", barwidth = grid::unit(2.2, "mm"), barheight = grid::unit(16, "mm")),
      size = ggplot2::guide_legend(title.position = "top", ncol = 1)
    ) +
    nature_theme(config$figure$base_font_pt, config$figure$font_family) +
    ggplot2::theme(
      strip.placement = "outside", strip.background = ggplot2::element_blank(),
      strip.text.y.left = ggplot2::element_text(angle = 0, face = "bold", size = 6.5,
                                                 margin = ggplot2::margin(r = 1.5)),
      panel.spacing.y = grid::unit(1.2, "mm"),
      axis.text.y = ggplot2::element_text(lineheight = .9),
      legend.position = "right", legend.box = "vertical", legend.box.just = "top",
      legend.spacing.y = grid::unit(1.2, "mm"), legend.margin = ggplot2::margin(0),
      legend.key.height = grid::unit(2.7, "mm"), legend.key.width = grid::unit(3, "mm")
    )
  add_panel_label(p, "a", config)
}

pathway_protein_links <- function(selected, mapping, included) {
  disease_scores <- dplyr::bind_rows(lapply(intersect(names(included), unique(selected$InputSet)), function(nm) {
    included[[nm]] |>
      dplyr::transmute(InputSet = nm, Protein = .data$Protein,
                       RecommendedBiomarkerScore = .data$RecommendedBiomarkerScore)
  }))
  mp <- dplyr::filter(mapping, .data$OriginalProtein %in% disease_scores$Protein, .data$MappingStatus == "mapped")
  selected |>
    tidyr::separate_rows(.data$SupportingProteins, sep = ";") |>
    dplyr::rename(EnrichmentID = .data$SupportingProteins) |>
    dplyr::inner_join(dplyr::select(mp, "OriginalProtein", "EnrichmentID"), by = "EnrichmentID") |>
    dplyr::transmute(InputSet = .data$InputSet, Database = .data$Database, TermID = .data$TermID,
                     TermName = .data$TermName, Protein = .data$OriginalProtein) |>
    dplyr::distinct() |>
    dplyr::inner_join(disease_scores, by = c("InputSet", "Protein")) |>
    dplyr::mutate(PathwayKey = paste(.data$InputSet, .data$Database, .data$TermID, sep = "|"),
                  CandidateKey = paste(.data$InputSet, .data$Protein, sep = "|"))
}

select_display_biomarkers_for_pathway_map <- function(links, max_biomarkers) {
  if (!nrow(links)) return(data.frame())
  candidates <- links |>
    dplyr::group_by(.data$CandidateKey, .data$InputSet, .data$Protein) |>
    dplyr::summarise(RecommendedBiomarkerScore = dplyr::first(.data$RecommendedBiomarkerScore),
                     NumberSelectedPathwaysSupported = dplyr::n_distinct(.data$PathwayKey), .groups = "drop")
  max_biomarkers <- min(as.integer(max_biomarkers), nrow(candidates))
  all_paths <- sort(unique(links$PathwayKey))
  selected <- character()
  coverage_selected <- character()
  if (nrow(candidates) <= max_biomarkers) {
    selected <- candidates$CandidateKey
  } else {
    if (max_biomarkers < length(all_paths)) {
      uncovered <- all_paths
      while (length(uncovered) && length(selected) < max_biomarkers) {
        eligible <- candidates[!candidates$CandidateKey %in% selected, , drop = FALSE]
        eligible$UncoveredPathways <- vapply(eligible$CandidateKey, function(key)
          length(intersect(uncovered, links$PathwayKey[links$CandidateKey == key])), integer(1))
        eligible$ScoreForOrder <- ifelse(is.na(eligible$RecommendedBiomarkerScore), -Inf, eligible$RecommendedBiomarkerScore)
        eligible <- eligible[order(-eligible$UncoveredPathways, -eligible$ScoreForOrder,
                                   -eligible$NumberSelectedPathwaysSupported, eligible$Protein, eligible$InputSet), ]
        if (!nrow(eligible) || eligible$UncoveredPathways[1] == 0) break
        chosen <- eligible$CandidateKey[1]
        selected <- c(selected, chosen); coverage_selected <- c(coverage_selected, chosen)
        uncovered <- setdiff(uncovered, links$PathwayKey[links$CandidateKey == chosen])
      }
    } else {
      # First cover each pathway. Within each uncovered pathway, score is primary;
      # multi-pathway support is secondary, and the protein identifier only breaks ties.
      for (path in all_paths) {
        if (any(links$CandidateKey %in% selected & links$PathwayKey == path)) next
        eligible <- candidates[candidates$CandidateKey %in% links$CandidateKey[links$PathwayKey == path], , drop = FALSE]
        eligible$ScoreForOrder <- ifelse(is.na(eligible$RecommendedBiomarkerScore), -Inf, eligible$RecommendedBiomarkerScore)
        eligible <- eligible[order(-eligible$ScoreForOrder, -eligible$NumberSelectedPathwaysSupported, eligible$Protein, eligible$InputSet), ]
        if (length(selected) < max_biomarkers && nrow(eligible)) {
          selected <- c(selected, eligible$CandidateKey[1]); coverage_selected <- c(coverage_selected, eligible$CandidateKey[1])
        }
      }
    }
    remaining <- candidates[!candidates$CandidateKey %in% selected, , drop = FALSE]
    remaining$ScoreForOrder <- ifelse(is.na(remaining$RecommendedBiomarkerScore), -Inf, remaining$RecommendedBiomarkerScore)
    remaining <- remaining[order(-remaining$ScoreForOrder, -remaining$NumberSelectedPathwaysSupported, remaining$Protein, remaining$InputSet), ]
    slots <- max_biomarkers - length(selected)
    if (slots > 0 && nrow(remaining)) selected <- c(selected, head(remaining$CandidateKey, slots))
  }
  candidates$Displayed <- candidates$CandidateKey %in% selected
  candidates$DisplaySelectionReason <- if (nrow(candidates) <= max_biomarkers) {
    "All candidates within display limit"
  } else ifelse(candidates$CandidateKey %in% coverage_selected, "Selected to preserve pathway coverage",
                ifelse(candidates$Displayed, "Selected by score after pathway coverage", "Not displayed: display limit"))
  candidates |>
    dplyr::select("InputSet", "Protein", "RecommendedBiomarkerScore", "NumberSelectedPathwaysSupported", "Displayed", "DisplaySelectionReason", "CandidateKey") |>
    dplyr::arrange(dplyr::desc(.data$Displayed), .data$InputSet,
                   dplyr::desc(dplyr::coalesce(.data$RecommendedBiomarkerScore, -Inf)),
                   dplyr::desc(.data$NumberSelectedPathwaysSupported), .data$Protein)
}

cubic_ribbon <- function(x0, x1, y0, y1, half_width, n = 32L) {
  t <- seq(0, 1, length.out = n)
  smooth <- 3 * t^2 - 2 * t^3
  x <- x0 + (x1 - x0) * t
  centre <- y0 + (y1 - y0) * smooth
  data.frame(x = c(x, rev(x)), y = c(centre + half_width, rev(centre - half_width)))
}

flow_strata <- function(items, key, degree, order_key, band, gap = .18) {
  items <- items[order(items[[order_key]], items[[key]]), , drop = FALSE]
  usable <- diff(band) - gap * max(0, nrow(items) - 1)
  unit_height <- usable / sum(items[[degree]])
  cursor <- band[1]
  items$ymin <- NA_real_; items$ymax <- NA_real_
  for (i in seq_len(nrow(items))) {
    items$ymin[i] <- cursor
    items$ymax[i] <- cursor + items[[degree]][i] * unit_height
    cursor <- items$ymax[i] + gap
  }
  items$y <- (items$ymin + items$ymax) / 2
  items$unit_height <- unit_height
  items
}

plot_xb <- function(links, config) {
  if (!nrow(links)) return(empty_panel("No significant pathway memberships", config))
  links <- links |> dplyr::distinct(.data$InputSet, .data$Protein, .data$TermName, .data$CandidateKey, .data$PathwayKey)
  disease_levels <- intersect(c("PD", "ALS"), unique(links$InputSet))
  bands <- list(PD = c(11, 21), ALS = c(0, 10))
  protein_parts <- list(); pathway_parts <- list()
  for (disease in disease_levels) {
    dl <- dplyr::filter(links, .data$InputSet == disease)
    path_order <- dl |>
      dplyr::distinct(.data$PathwayKey, .data$TermName) |>
      dplyr::arrange(.data$TermName) |>
      dplyr::mutate(PathOrder = dplyr::row_number())
    protein_order <- dl |>
      dplyr::left_join(path_order, by = c("PathwayKey", "TermName")) |>
      dplyr::group_by(.data$CandidateKey, .data$Protein) |>
      dplyr::summarise(Order = mean(.data$PathOrder), Degree = dplyr::n_distinct(.data$PathwayKey), .groups = "drop")
    pathway_order <- dl |>
      dplyr::count(.data$PathwayKey, .data$TermName, name = "Degree") |>
      dplyr::left_join(path_order, by = c("PathwayKey", "TermName")) |>
      dplyr::rename(Order = "PathOrder")
    protein_parts[[disease]] <- flow_strata(protein_order, "CandidateKey", "Degree", "Order", bands[[disease]]) |>
      dplyr::mutate(InputSet = disease)
    pathway_parts[[disease]] <- flow_strata(pathway_order, "PathwayKey", "Degree", "Order", bands[[disease]]) |>
      dplyr::mutate(InputSet = disease)
  }
  proteins <- dplyr::bind_rows(protein_parts)
  pathways <- dplyr::bind_rows(pathway_parts)
  links <- links |>
    dplyr::left_join(dplyr::select(proteins, "CandidateKey", ProteinYmin = "ymin", ProteinUnit = "unit_height"), by = "CandidateKey") |>
    dplyr::left_join(dplyr::select(pathways, "PathwayKey", PathwayYmin = "ymin", PathwayUnit = "unit_height"), by = "PathwayKey") |>
    dplyr::group_by(.data$CandidateKey) |>
    dplyr::arrange(.data$PathwayYmin, .data$PathwayKey, .by_group = TRUE) |>
    dplyr::mutate(ProteinSlot = dplyr::row_number()) |>
    dplyr::ungroup() |>
    dplyr::group_by(.data$PathwayKey) |>
    dplyr::arrange(.data$ProteinYmin, .data$CandidateKey, .by_group = TRUE) |>
    dplyr::mutate(PathwaySlot = dplyr::row_number()) |>
    dplyr::ungroup() |>
    dplyr::mutate(
      y0 = .data$ProteinYmin + (.data$ProteinSlot - .5) * .data$ProteinUnit,
      y1 = .data$PathwayYmin + (.data$PathwaySlot - .5) * .data$PathwayUnit,
      FlowID = dplyr::row_number()
    )
  ribbons <- dplyr::bind_rows(lapply(seq_len(nrow(links)), function(i) {
    z <- cubic_ribbon(.20, .80, links$y0[i], links$y1[i], min(links$ProteinUnit[i], links$PathwayUnit[i]) * .34)
    z$FlowID <- links$FlowID[i]; z$InputSet <- links$InputSet[i]; z
  }))
  p <- ggplot2::ggplot() +
    ggplot2::geom_polygon(data = ribbons, ggplot2::aes(.data$x, .data$y, group = .data$FlowID, fill = .data$InputSet),
                          alpha = .25, colour = NA) +
    ggplot2::geom_rect(data = proteins, ggplot2::aes(xmin = .075, xmax = .20, ymin = .data$ymin, ymax = .data$ymax, fill = .data$InputSet),
                       colour = "white", linewidth = .2) +
    ggplot2::geom_rect(data = pathways, ggplot2::aes(xmin = .80, xmax = .925, ymin = .data$ymin, ymax = .data$ymax, fill = .data$InputSet),
                       colour = "white", linewidth = .2) +
    ggplot2::geom_text(data = proteins, ggplot2::aes(x = .06, y = .data$y, label = .data$Protein),
                       hjust = 1, size = 5 / ggplot2::.pt, family = config$figure$font_family) +
    ggplot2::geom_text(data = pathways, ggplot2::aes(x = .94, y = .data$y, label = stringr::str_wrap(.data$TermName, 25)),
                       hjust = 0, lineheight = .9, size = 5 / ggplot2::.pt, family = config$figure$font_family) +
    ggplot2::annotate("text", x = .1375, y = 22, label = "RECOMMENDED\nBIOMARKERS", fontface = "bold", lineheight = .9,
                      size = 5.5 / ggplot2::.pt, family = config$figure$font_family) +
    ggplot2::annotate("text", x = .8625, y = 22, label = "ENRICHED\nPATHWAYS", fontface = "bold", lineheight = .9,
                      size = 5.5 / ggplot2::.pt, family = config$figure$font_family) +
    ggplot2::scale_fill_manual(values = c(PD = COLORS$PD, ALS = COLORS$ALS), breaks = c("PD", "ALS"), name = NULL) +
    ggplot2::coord_cartesian(xlim = c(-.10, 1.48), ylim = c(-.5, 22.5), clip = "off") +
    ggplot2::theme_void(base_family = config$figure$font_family, base_size = config$figure$base_font_pt) +
    ggplot2::theme(
      legend.position = "bottom", legend.direction = "horizontal",
      legend.text = ggplot2::element_text(size = 5.5), legend.key.size = grid::unit(3, "mm"),
      legend.margin = ggplot2::margin(0), plot.margin = grid::unit(c(1, 18, 1, 12), "mm")
    )
  add_panel_label(p, "b", config)
}

correlation_plot_data <- function(result, analysis) {
  if (!length(result$correlation)) return(data.frame())
  d <- as.data.frame(as.table(result$correlation), stringsAsFactors = FALSE)
  names(d) <- c("ProteinX", "ProteinY", "Rho")
  d$Analysis <- analysis; d
}

correlation_hclust <- function(correlation, config) {
  if (nrow(correlation) < 2) return(FALSE)
  x <- correlation; x[x < -1] <- -1; x[x > 1] <- 1; x[is.na(x)] <- 0; diag(x) <- 1
  distance <- 1 - x; distance[distance < 0] <- 0
  stats::hclust(stats::as.dist(distance), method = config$correlation$linkage %||% "complete")
}

plot_xc <- function(correlations, config) {
  if (!requireNamespace("ComplexHeatmap", quietly = TRUE)) stopf("Figure X-C requires ComplexHeatmap")
  if (!requireNamespace("circlize", quietly = TRUE)) stopf("Figure X-C requires circlize")
  valid <- correlations[vapply(correlations, function(x) length(x$correlation) > 0, logical(1))]
  if (!length(valid)) stopf("Figure X-C requires at least two measured biomarkers per disease cohort")
  colour_fun <- circlize::colorRamp2(c(-1, 0, 1), unname(CORRELATION_COLOURS))
  largest_matrix <- max(vapply(valid, function(x) max(nrow(x$correlation), ncol(x$correlation)), numeric(1)))
  cell_mm <- min(6.5, 58 / largest_matrix)
  heatmaps <- lapply(names(valid), function(nm) {
    mat <- valid[[nm]]$correlation
    row_cluster <- if (isTRUE(config$correlation$cluster_rows)) correlation_hclust(mat, config) else FALSE
    column_cluster <- if (isTRUE(config$correlation$cluster_columns)) correlation_hclust(mat, config) else FALSE
    ComplexHeatmap::Heatmap(mat, name = paste0("rho_", nm), col = colour_fun, na_col = "#E6E6E6",
      cluster_rows = row_cluster, cluster_columns = column_cluster,
      row_names_gp = grid::gpar(fontsize = 5.2, fontfamily = config$figure$font_family),
      row_names_side = if (identical(nm, names(valid)[1])) "left" else "right",
      column_names_gp = grid::gpar(fontsize = 5.2, fontfamily = config$figure$font_family),
      column_names_rot = 45, column_title = sprintf("%s (n = %d)", nm, valid[[nm]]$n),
      column_title_gp = grid::gpar(fontsize = 6, fontface = "bold", fontfamily = config$figure$font_family),
      heatmap_legend_param = list(
        title = "Spearman rho", at = c(-1, 0, 1), direction = "horizontal",
        title_position = "topcenter", legend_width = grid::unit(28, "mm"),
        title_gp = grid::gpar(fontsize = 6), labels_gp = grid::gpar(fontsize = 5)
      ),
      row_dend_width = grid::unit(7, "mm"), column_dend_height = grid::unit(7, "mm"),
      width = grid::unit(ncol(mat) * cell_mm, "mm"), height = grid::unit(nrow(mat) * cell_mm, "mm"),
      rect_gp = grid::gpar(col = "white", lwd = .35),
      show_heatmap_legend = identical(nm, names(valid)[length(valid)]), border = TRUE,
      use_raster = FALSE
    )
  })
  Reduce(`+`, heatmaps)
}

plot_xc_effect_concordance <- function(concordance, statistics, config) {
  d <- dplyr::filter(concordance, .data$BothEffectsEstimable)
  if (!nrow(d)) return(empty_panel("No proteins have estimable PD and ALS effects", config))
  recommended <- dplyr::filter(d, .data$Recommended)
  background <- dplyr::filter(d, !.data$Recommended)
  stat_label <- sprintf(
    "Spearman rho = %s; n = %d\nP = %s",
    ifelse(is.finite(statistics$SpearmanRho), formatC(statistics$SpearmanRho, digits = 2, format = "f"), "NA"),
    statistics$NProteins,
    ifelse(is.finite(statistics$PValue), format.pval(statistics$PValue, digits = 2, eps = 1e-4), "NA")
  )
  p <- ggplot2::ggplot() +
    ggplot2::geom_hline(yintercept = 0, colour = "#9AA1A9", linewidth = .3, linetype = "dashed") +
    ggplot2::geom_vline(xintercept = 0, colour = "#9AA1A9", linewidth = .3, linetype = "dashed") +
    ggplot2::geom_point(data = background, ggplot2::aes(.data$PD_HedgesG, .data$ALS_HedgesG),
                        colour = "#9AA1A9", alpha = .28, size = 1.25) +
    ggplot2::geom_point(data = recommended,
      ggplot2::aes(.data$PD_HedgesG, .data$ALS_HedgesG, colour = .data$BiomarkerClass,
                   shape = .data$DirectionClass), size = 2.5, stroke = .55) +
    ggplot2::annotate("text", x = -Inf, y = Inf, label = stat_label, hjust = -.05, vjust = 1.1,
                      size = 5.5 / ggplot2::.pt, family = config$figure$font_family, lineheight = .95) +
    ggplot2::scale_colour_manual(values = BIOMARKER_CLASS_COLOURS, drop = FALSE, name = "Biomarker class") +
    ggplot2::scale_shape_manual(values = DIRECTION_CLASS_SHAPES, drop = FALSE, name = "Direction class") +
    ggplot2::labs(x = "PD Hedges' g", y = "ALS Hedges' g") +
    ggplot2::coord_equal(clip = "off") +
    nature_theme(config$figure$base_font_pt, config$figure$font_family) +
    ggplot2::theme(
      panel.grid.major = ggplot2::element_line(colour = COLORS$Grid, linewidth = .25),
      panel.grid.minor = ggplot2::element_blank(),
      legend.position = "right", legend.box = "vertical",
      legend.text = ggplot2::element_text(size = 5), legend.title = ggplot2::element_text(size = 5.5),
      plot.margin = grid::unit(c(1, 2, 1, 2), "mm")
    )
  labels <- dplyr::filter(recommended, .data$DisplayLabel)
  if (nrow(labels)) {
    if (requireNamespace("ggrepel", quietly = TRUE)) {
      p <- p + ggrepel::geom_text_repel(
        data = labels, ggplot2::aes(.data$PD_HedgesG, .data$ALS_HedgesG, label = .data$Protein,
                                    colour = .data$BiomarkerClass),
        size = 5 / ggplot2::.pt, family = config$figure$font_family,
        box.padding = .25, point.padding = .15, min.segment.length = 0,
        segment.size = .25, max.overlaps = Inf, show.legend = FALSE, seed = config$project$seed
      )
    } else {
      p <- p + ggplot2::geom_text(data = labels,
        ggplot2::aes(.data$PD_HedgesG, .data$ALS_HedgesG, label = .data$Protein),
        nudge_y = .05, size = 5 / ggplot2::.pt, family = config$figure$font_family, show.legend = FALSE)
    }
  }
  add_panel_label(p, "c", config)
}

empty_panel <- function(message, config) add_panel_label(ggplot2::ggplot() + ggplot2::annotate("text", x = .5, y = .5, label = message, size = 2.5) + ggplot2::theme_void(), "", config)

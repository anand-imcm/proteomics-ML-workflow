upset_combination_table <- function(membership) {
  membership |>
    dplyr::mutate(Combination = paste0(ifelse(.data$PD, "PD", ""), ifelse(.data$ALS, "+ALS", ""),
                                        ifelse(.data$CrossDisease, "+CrossDisease", "")),
                  Combination = sub("^\\+", "", .data$Combination)) |>
    dplyr::count(.data$PD, .data$ALS, .data$CrossDisease, .data$Combination, name = "IntersectionSize") |>
    dplyr::arrange(dplyr::desc(.data$IntersectionSize), .data$Combination)
}

plot_ya <- function(membership, config) {
  if (!requireNamespace("ComplexUpset", quietly = TRUE)) stopf("Figure Y-A requires ComplexUpset")
  upset_themes <- getFromNamespace("upset_themes", "ComplexUpset")
  upset_themes <- lapply(upset_themes, function(theme_list) lapply(theme_list, function(theme_part) {
    theme_part[["legend.text.align"]] <- NULL
    theme_part[["legend.title.align"]] <- NULL
    theme_part
  }))
  p <- ComplexUpset::upset(membership, intersect = c("PD", "ALS", "CrossDisease"), name = "",
    width_ratio = 0.24, height_ratio = 0.75,
    themes = upset_themes,
    base_annotations = list("Intersection size" = ComplexUpset::intersection_size(counts = TRUE,
      text = list(size = 5 / ggplot2::.pt), width = .62, fill = "#465362")),
    matrix = ComplexUpset::intersection_matrix(
      geom = ggplot2::geom_point(size = 2.2),
      segment = ggplot2::geom_segment(linewidth = .45)
    ),
    set_sizes = ComplexUpset::upset_set_size(geom = ggplot2::geom_bar(width = .62, fill = COLORS$Neutral))) &
    ggplot2::theme(text = ggplot2::element_text(size = config$figure$base_font_pt, family = config$figure$font_family),
                   axis.text = ggplot2::element_text(size = 5.5, colour = "#202124"),
                   axis.title = ggplot2::element_text(size = 6),
                   plot.margin = grid::unit(c(1, 1, 1, 1), "mm"))
  add_panel_label(p, "a", config)
}

plot_yb <- function(score_matrix, config) {
  if (!requireNamespace("ComplexHeatmap", quietly = TRUE)) stopf("Figure Y-B requires ComplexHeatmap")
  if (!requireNamespace("circlize", quietly = TRUE)) stopf("Figure Y-B requires circlize")
  if (!nrow(score_matrix) || all(is.na(as.matrix(score_matrix[c("PD", "ALS", "CrossDisease")]))) ) stopf("No proteins meet the configured Y-B support rule")
  mat <- as.matrix(score_matrix[c("PD", "ALS", "CrossDisease")]); rownames(mat) <- score_matrix$Protein
  observed <- mat[is.finite(mat)]
  limits <- if (length(observed) && min(observed) >= 0 && max(observed) <= 1) c(0, .5, 1) else c(min(observed), stats::median(observed), max(observed))
  if (length(unique(limits)) < 3) limits <- seq(min(observed) - .001, max(observed) + .001, length.out = 3)
  biomarker_levels <- unique(score_matrix$BiomarkerClass)
  direction_levels <- unique(score_matrix$DirectionClass)
  biomarker_colours <- BIOMARKER_CLASS_COLOURS[biomarker_levels]
  direction_colours <- DIRECTION_CLASS_COLOURS[direction_levels]
  readable_class <- function(x) gsub("_", " ", x, fixed = TRUE)
  annotation <- ComplexHeatmap::rowAnnotation(
    BiomarkerClass = score_matrix$BiomarkerClass,
    DirectionClass = score_matrix$DirectionClass,
    col = list(BiomarkerClass = biomarker_colours, DirectionClass = direction_colours),
    show_annotation_name = FALSE,
    width = grid::unit(5.2, "mm"), simple_anno_size = grid::unit(2.4, "mm"),
    annotation_legend_param = list(
      BiomarkerClass = list(at = biomarker_levels, labels = readable_class(biomarker_levels),
                            labels_gp = grid::gpar(fontsize = 5), title_gp = grid::gpar(fontsize = 6)),
      DirectionClass = list(at = direction_levels, labels = readable_class(direction_levels),
                            labels_gp = grid::gpar(fontsize = 5), title_gp = grid::gpar(fontsize = 6))
    )
  )
  annotation + ComplexHeatmap::Heatmap(mat, name = "Recommended\nBiomarkerScore\n(grey = NA)",
    col = circlize::colorRamp2(limits, c("#F7FBFF", "#6BAED6", "#08306B")), na_col = COLORS$Missing,
    cluster_rows = FALSE, cluster_columns = FALSE, border = TRUE,
    row_names_gp = grid::gpar(fontsize = 5.2, fontfamily = config$figure$font_family),
    column_names_gp = grid::gpar(fontsize = 5.5, fontfamily = config$figure$font_family),
    width = grid::unit(ncol(mat) * 5.6, "mm"), height = grid::unit(nrow(mat) * 5.6, "mm"),
    rect_gp = grid::gpar(col = "white", lwd = .45), use_raster = FALSE,
    heatmap_legend_param = list(
      direction = "horizontal", title_position = "topcenter", legend_width = grid::unit(25, "mm"),
      title_gp = grid::gpar(fontsize = 6), labels_gp = grid::gpar(fontsize = 5)
    ))
}

plot_yc <- function(nonredundant, config) {
  d <- select_terms(nonredundant, config$visualization$max_pathways_YC, config$enrichment$fdr_threshold)
  if (!nrow(d)) return(empty_panel("No significant enrichment in derived sets", config))
  preferred_levels <- c("Shared", "PD_distinct", "ALS_distinct", "Shared_differential",
                        "PD_selective", "ALS_selective", "Differential_only")
  column_levels <- preferred_levels[preferred_levels %in% unique(d$InputSet)]
  column_labels <- setNames(gsub("_", "\n", column_levels, fixed = TRUE), column_levels)
  term_order <- d |>
    dplyr::group_by(.data$TermName) |>
    dplyr::summarise(BestFDR = min(.data$AdjustedPValue), .groups = "drop") |>
    dplyr::arrange(dplyr::desc(.data$BestFDR), .data$TermName) |>
    dplyr::pull(.data$TermName)
  matrix_data <- tidyr::expand_grid(TermName = term_order, InputSet = column_levels) |>
    dplyr::left_join(dplyr::select(d, "TermName", "InputSet", "GeneCount", "AdjustedPValue"),
                     by = c("TermName", "InputSet")) |>
    dplyr::mutate(
      InputSet = factor(.data$InputSet, levels = column_levels),
      TermName = factor(.data$TermName, levels = term_order)
    )
  p <- ggplot2::ggplot(matrix_data, ggplot2::aes(.data$InputSet, .data$TermName)) +
    ggplot2::geom_tile(fill = "white", colour = COLORS$Grid, linewidth = .3) +
    ggplot2::geom_point(
      data = dplyr::filter(matrix_data, !is.na(.data$AdjustedPValue)),
      ggplot2::aes(size = .data$GeneCount, fill = safe_neglog10(.data$AdjustedPValue)),
      shape = 21, colour = "#344054", stroke = .25
    ) +
    scale_significance_fill(name = expression(-log[10](FDR))) +
    ggplot2::scale_size_area(max_size = 4, name = "Count") +
    ggplot2::scale_x_discrete(labels = column_labels, expand = c(0, 0)) +
    ggplot2::scale_y_discrete(labels = function(x) stringr::str_wrap(x, 28), expand = c(0, 0)) +
    ggplot2::labs(x = NULL, y = NULL) +
    ggplot2::guides(
      fill = ggplot2::guide_colourbar(title.position = "top", barwidth = grid::unit(16, "mm"), barheight = grid::unit(2.2, "mm")),
      size = ggplot2::guide_legend(title.position = "top", nrow = 1)
    ) +
    nature_theme(config$figure$base_font_pt, config$figure$font_family) +
    ggplot2::theme(
      axis.line = ggplot2::element_blank(), axis.ticks = ggplot2::element_blank(),
      axis.text.x = ggplot2::element_text(face = "bold", size = 5.2, angle = 28, hjust = 1,
                                          margin = ggplot2::margin(t = 1.5)),
      axis.text.y = ggplot2::element_text(lineheight = .9),
      legend.position = "bottom", legend.box = "vertical", legend.box.just = "left", legend.margin = ggplot2::margin(0),
      plot.margin = grid::unit(c(1, 2, 1, 1.5), "mm")
    )
  add_panel_label(p, "c", config)
}

plot_yd <- function(edges, config) {
  if (!nrow(edges)) return(empty_panel("No shared pathways with PD- and ALS-supported proteins", config))
  if (!"BiomarkerClass" %in% names(edges)) stopf("Y-D requires current BiomarkerClass metadata")
  node_class <- edges |>
    dplyr::transmute(Protein = .data$Protein, BiomarkerClass = .data$BiomarkerClass) |>
    dplyr::distinct() |>
    dplyr::mutate(NodeClass = dplyr::case_when(
      .data$BiomarkerClass %in% c("Shared", "Shared_differential") ~ "Shared protein",
      .data$BiomarkerClass %in% c("PD_selective", "PD_distinct") ~ "PD-axis protein",
      .data$BiomarkerClass %in% c("ALS_selective", "ALS_distinct") ~ "ALS-axis protein",
      TRUE ~ "Other protein")) |>
    dplyr::filter(.data$NodeClass != "Other protein")
  column_x <- c("PD-axis protein" = -1.2, "Shared protein" = -.40,
                "Pathway" = .40, "ALS-axis protein" = 1.2)
  column_counts <- c(
    sum(node_class$NodeClass == "PD-axis protein"),
    sum(node_class$NodeClass == "Shared protein"),
    length(unique(edges$Pathway)),
    sum(node_class$NodeClass == "ALS-axis protein")
  )
  y_span <- max(1, .32 * (max(column_counts) - 1))
  pathways <- data.frame(
    name = sort(unique(edges$Pathway)), NodeClass = "Pathway", x = column_x[["Pathway"]],
    y = if (length(unique(edges$Pathway)) == 1) 0 else seq(from = y_span, to = -y_span,
                                                            length.out = length(unique(edges$Pathway))),
    stringsAsFactors = FALSE
  )
  target_y <- setNames(pathways$y, pathways$name)
  place_column <- function(class, x) {
    z <- node_class |>
      dplyr::filter(.data$NodeClass == class) |>
      dplyr::left_join(edges |> dplyr::transmute(Protein = .data$Source, Pathway = .data$Target), by = "Protein") |>
      dplyr::group_by(.data$Protein, .data$NodeClass) |>
      dplyr::summarise(TargetMean = mean(target_y[.data$Pathway]), .groups = "drop") |>
      dplyr::arrange(dplyr::desc(.data$TargetMean), .data$Protein)
    if (!nrow(z)) return(z |> dplyr::mutate(name = character(), x = numeric(), y = numeric()))
    z |>
      dplyr::mutate(name = .data$Protein, x = x,
                    y = if (dplyr::n() == 1) 0 else seq(from = y_span, to = -y_span, length.out = dplyr::n()))
  }
  pd_nodes <- place_column("PD-axis protein", column_x[["PD-axis protein"]])
  shared_nodes <- place_column("Shared protein", column_x[["Shared protein"]])
  als_nodes <- place_column("ALS-axis protein", column_x[["ALS-axis protein"]])
  protein_nodes <- dplyr::bind_rows(pd_nodes, als_nodes, shared_nodes) |>
    dplyr::select("name", "Protein", "NodeClass", "x", "y")
  all_nodes <- dplyr::bind_rows(
    protein_nodes |> dplyr::select("name", "NodeClass", "x", "y"),
    pathways
  )
  edge_data <- edges |>
    dplyr::distinct(.data$Source, .data$Target) |>
    dplyr::left_join(dplyr::select(all_nodes, Source = "name", x, y, SourceClass = "NodeClass"), by = "Source") |>
    dplyr::left_join(dplyr::select(pathways, Target = "name", yend = y), by = "Target") |>
    dplyr::mutate(xend = dplyr::case_when(
      .data$SourceClass == "ALS-axis protein" ~ column_x[["Pathway"]] + .13,
      TRUE ~ column_x[["Pathway"]] - .13
    ))
  p <- ggplot2::ggplot() +
    ggplot2::geom_curve(data = dplyr::filter(edge_data, .data$SourceClass == "PD-axis protein"),
      ggplot2::aes(x = .data$x, y = .data$y, xend = .data$xend, yend = .data$yend),
      curvature = .08, colour = COLORS$PD, alpha = .32, linewidth = .38) +
    ggplot2::geom_curve(data = dplyr::filter(edge_data, .data$SourceClass == "ALS-axis protein"),
      ggplot2::aes(x = .data$x, y = .data$y, xend = .data$xend, yend = .data$yend),
      curvature = -.08, colour = COLORS$ALS, alpha = .32, linewidth = .38) +
    ggplot2::geom_curve(data = dplyr::filter(edge_data, .data$SourceClass == "Shared protein"),
      ggplot2::aes(x = .data$x, y = .data$y, xend = .data$xend, yend = .data$yend),
      curvature = .06, colour = COLORS$Shared, alpha = .32, linewidth = .38) +
    ggplot2::geom_point(data = protein_nodes,
      ggplot2::aes(.data$x, .data$y, fill = .data$NodeClass, shape = .data$NodeClass),
      size = 2.6, colour = "#30343B", stroke = .45) +
    ggplot2::geom_point(data = pathways, ggplot2::aes(.data$x, .data$y, fill = .data$NodeClass, shape = .data$NodeClass),
      size = 3.2, colour = "#30343B", stroke = .5) +
    ggplot2::geom_label(data = pathways,
      ggplot2::aes(.data$x, .data$y, label = stringr::str_wrap(.data$name, 21)),
      size = 5 / ggplot2::.pt, family = config$figure$font_family, lineheight = .88,
      label.size = .2, label.padding = grid::unit(.7, "mm"), fill = "white", colour = "#202124") +
    ggplot2::geom_text(data = pd_nodes, ggplot2::aes(x = .data$x - .065, y = .data$y, label = .data$Protein),
      hjust = 1, size = 5 / ggplot2::.pt, family = config$figure$font_family) +
    ggplot2::geom_text(data = shared_nodes, ggplot2::aes(x = .data$x - .065, y = .data$y, label = .data$Protein),
      hjust = 1, size = 5 / ggplot2::.pt, family = config$figure$font_family) +
    ggplot2::geom_text(data = als_nodes, ggplot2::aes(x = .data$x + .055, y = .data$y, label = .data$Protein),
      hjust = 0, size = 5 / ggplot2::.pt, family = config$figure$font_family) +
    ggplot2::annotate("text", x = column_x[["PD-axis protein"]], y = y_span + .34, label = "PD-SELECTIVE /\nPD-DISTINCT", fontface = "bold", colour = COLORS$PD,
      lineheight = .9,
      size = 5.5 / ggplot2::.pt, family = config$figure$font_family) +
    ggplot2::annotate("text", x = column_x[["Shared protein"]], y = y_span + .34, label = "SHARED\nPROTEINS", fontface = "bold", colour = COLORS$Shared,
      lineheight = .9, size = 5.5 / ggplot2::.pt, family = config$figure$font_family) +
    ggplot2::annotate("text", x = column_x[["Pathway"]], y = y_span + .34, label = "SHARED\nPATHWAYS", fontface = "bold", colour = "#4A4A4A",
      lineheight = .9, size = 5.5 / ggplot2::.pt, family = config$figure$font_family) +
    ggplot2::annotate("text", x = column_x[["ALS-axis protein"]], y = y_span + .34, label = "ALS-SELECTIVE /\nALS-DISTINCT", fontface = "bold", colour = COLORS$ALS,
      lineheight = .9,
      size = 5.5 / ggplot2::.pt, family = config$figure$font_family) +
    ggplot2::scale_fill_manual(values = c("PD-axis protein" = COLORS$PD, "ALS-axis protein" = COLORS$ALS,
      "Shared protein" = COLORS$Shared, "Pathway" = "white"),
      breaks = c("PD-axis protein", "ALS-axis protein", "Shared protein", "Pathway"), name = NULL) +
    ggplot2::scale_shape_manual(values = c("PD-axis protein" = 21, "ALS-axis protein" = 22,
      "Shared protein" = 23, "Pathway" = 22),
      breaks = c("PD-axis protein", "ALS-axis protein", "Shared protein", "Pathway"), name = NULL) +
    ggplot2::guides(fill = ggplot2::guide_legend(nrow = 1, byrow = TRUE), shape = ggplot2::guide_legend(nrow = 1, byrow = TRUE)) +
    ggplot2::coord_cartesian(xlim = c(-1.65, 1.65), ylim = c(-y_span - .25, y_span + .58), clip = "off") +
    ggplot2::theme_void(base_family = config$figure$font_family, base_size = config$figure$base_font_pt) +
    ggplot2::theme(legend.position = "bottom", legend.text = ggplot2::element_text(size = 5),
      legend.key.size = grid::unit(3, "mm"), legend.margin = ggplot2::margin(0),
      plot.margin = grid::unit(c(2, 8, 1, 8), "mm"))
  add_panel_label(p, "d", config)
}

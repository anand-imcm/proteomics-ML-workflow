COLORS <- list(
  PD = "#0072B2",
  ALS = "#D55E00",
  Shared = "#009E73",
  CrossDisease = "#CC79A7",
  Neutral = "#66717E",
  Grid = "#E4E7EB",
  Missing = "#B8BDC5"
)

SIGNIFICANCE_COLOURS <- c("#EFF3FF", "#BDD7E7", "#6BAED6", "#2171B5", "#08306B")
CORRELATION_COLOURS <- c(negative = "#2166AC", zero = "#F7F7F7", positive = "#B2182B")
BIOMARKER_CLASS_COLOURS <- c(
  Shared = COLORS$Shared,
  Shared_differential = "#6A51A3",
  PD_selective = "#56B4E9",
  PD_distinct = COLORS$PD,
  ALS_selective = "#E69F00",
  ALS_distinct = COLORS$ALS,
  Differential_only = COLORS$CrossDisease
)
DIRECTION_CLASS_SHAPES <- c(
  Concordant_up = 24,
  Concordant_down = 25,
  Discordant_PD_up_ALS_down = 23,
  Discordant_PD_down_ALS_up = 22,
  Direction_uncertain = 21
)
DIRECTION_CLASS_COLOURS <- c(
  Concordant_up = "#B2182B",
  Concordant_down = "#2166AC",
  Discordant_PD_up_ALS_down = "#7B3294",
  Discordant_PD_down_ALS_up = "#008837",
  Direction_uncertain = COLORS$Missing
)

nature_theme <- function(base_size = 6, family = "Arial") {
  ggplot2::theme_classic(base_size = base_size, base_family = family) +
    ggplot2::theme(
      axis.title = ggplot2::element_text(size = base_size + 0.5, colour = "#202124"),
      axis.text = ggplot2::element_text(size = base_size, colour = "#202124"),
      axis.line = ggplot2::element_line(linewidth = 0.25, colour = "#4A4A4A"),
      axis.ticks = ggplot2::element_line(linewidth = 0.25, colour = "#4A4A4A"),
      legend.title = ggplot2::element_text(size = base_size),
      legend.text = ggplot2::element_text(size = base_size),
      legend.key.height = grid::unit(3, "mm"),
      legend.key.width = grid::unit(4, "mm"),
      plot.margin = grid::unit(c(1.5, 2, 1.5, 1.5), "mm"),
      plot.title = ggplot2::element_blank()
    )
}

scale_significance_colour <- function(...) {
  ggplot2::scale_colour_gradientn(colours = SIGNIFICANCE_COLOURS, ...)
}

scale_significance_fill <- function(...) {
  ggplot2::scale_fill_gradientn(colours = SIGNIFICANCE_COLOURS, ...)
}

add_panel_label <- function(p, label, config) {
  if (!requireNamespace("patchwork", quietly = TRUE)) stopf("patchwork is required for panel framing")
  label_strip <- ggplot2::ggplot() +
    ggplot2::annotate(
      "text", x = 0, y = 0.5, label = label, hjust = 0, vjust = 0.5,
      size = config$figure$panel_label_pt / ggplot2::.pt,
      fontface = "bold", family = config$figure$font_family, colour = "#111111"
    ) +
    ggplot2::coord_cartesian(xlim = c(0, 1), ylim = c(0, 1), clip = "off") +
    ggplot2::theme_void() +
    ggplot2::theme(plot.margin = grid::unit(c(0, 0, 0, 0.8), "mm"))
  label_strip / patchwork::wrap_elements(full = p) +
    patchwork::plot_layout(heights = c(0.06, 0.94))
}

safe_neglog10 <- function(p) -log10(pmax(p, .Machine$double.xmin))

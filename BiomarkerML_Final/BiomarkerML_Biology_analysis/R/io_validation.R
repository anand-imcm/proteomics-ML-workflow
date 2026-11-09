# Immutable upstream input reading and schema validation.

`%||%` <- function(x, y) if (is.null(x)) y else x

stopf <- function(...) stop(sprintf(...), call. = FALSE)

read_csv_strict <- function(path) {
  if (!file.exists(path)) stopf("Required input does not exist: %s", path)
  out <- readr::read_csv(path, show_col_types = FALSE, name_repair = "minimal")
  if (!nrow(out)) stopf("Input has no rows: %s", path)
  out
}

normalise_logical <- function(x, field, path) {
  if (is.logical(x)) return(x)
  val <- tolower(trimws(as.character(x)))
  allowed <- c("true", "false", "t", "f", "1", "0", "yes", "no")
  if (any(!is.na(val) & !val %in% allowed))
    stopf("%s contains non-logical values in %s", field, path)
  val %in% c("true", "t", "1", "yes")
}

read_recommended_biomarkers <- function(path, analysis_name) {
  x <- read_csv_strict(path)
  required <- c("Protein", "Recommended", "RecommendedBiomarkerScore")
  missing <- setdiff(required, names(x))
  if (length(missing)) stopf("%s is missing required column(s): %s", path, paste(missing, collapse = ", "))
  x$Protein <- trimws(as.character(x$Protein))
  if (anyNA(x$Protein) || any(x$Protein == "")) stopf("Protein contains missing/empty values in %s", path)
  if (anyDuplicated(x$Protein)) stopf("Protein identifiers must be unique in %s", path)
  x$Recommended <- normalise_logical(x$Recommended, "Recommended", path)
  x$RecommendedBiomarkerScore <- suppressWarnings(as.numeric(x$RecommendedBiomarkerScore))
  # Scores may be NA upstream; retain them verbatim (never impute or reconstruct).
  x$Analysis <- analysis_name
  x
}

included_biomarkers <- function(x, config) {
  if (!isTRUE(config$biomarker_selection$use_recommended_only))
    stopf("This pipeline intentionally supports only Recommended == TRUE inclusion.")
  dplyr::filter(x, .data$Recommended %in% TRUE)
}

detect_expression_layout <- function(x, config) {
  required_long <- c(config$project$expression_sample_id_column, config$project$expression_protein_id_column,
                     config$project$expression_value_column)
  if (all(required_long %in% names(x))) return("long")
  "wide"
}

read_expression_matrix <- function(path, config) {
  x <- read_csv_strict(path)
  layout <- detect_expression_layout(x, config)
  sample_col <- config$project$expression_sample_id_column
  protein_col <- config$project$expression_protein_id_column
  group_col <- config$project$expression_group_column
  value_col <- config$project$expression_value_column
  if (layout == "long") {
    x <- dplyr::transmute(x, SampleID = as.character(.data[[sample_col]]), Protein = as.character(.data[[protein_col]]),
                          Abundance = suppressWarnings(as.numeric(.data[[value_col]])),
                          Group = if (group_col %in% names(x)) as.character(.data[[group_col]]) else NA_character_)
  } else {
    if (!sample_col %in% names(x)) stopf("Wide expression matrix must contain sample ID column '%s'", sample_col)
    metadata <- intersect(c(sample_col, group_col), names(x))
    protein_cols <- setdiff(names(x), metadata)
    if (!length(protein_cols)) stopf("No protein columns found in wide expression matrix")
    x <- tidyr::pivot_longer(x, cols = dplyr::all_of(protein_cols), names_to = "Protein", values_to = "Abundance")
    x <- dplyr::transmute(x, SampleID = as.character(.data[[sample_col]]), Protein = as.character(.data$Protein),
                          Abundance = suppressWarnings(as.numeric(.data$Abundance)),
                          Group = if (group_col %in% names(x)) as.character(.data[[group_col]]) else NA_character_)
  }
  x$Protein <- trimws(x$Protein)
  if (!nrow(x) || !any(is.finite(x$Abundance))) stopf("Expression matrix contains no finite abundance values")
  if (anyDuplicated(x[c("SampleID", "Protein")])) stopf("Expression data have duplicate SampleID/Protein pairs")
  attr(x, "layout") <- layout
  x
}

input_paths <- function(root = ".") list(
  PD = file.path(root, "input", "PD", "recommended_biomarkers.csv"),
  ALS = file.path(root, "input", "ALS", "recommended_biomarkers.csv"),
  CrossDisease = file.path(root, "input", "CrossDisease", "recommended_biomarkers.csv"),
  expression = file.path(root, "input", "expression_matrix.csv")
)

read_config <- function(path = "config/analysis_config.yaml") {
  if (!file.exists(path)) stopf("Configuration file not found: %s", path)
  yaml::read_yaml(path)
}

packages <- c(
  "graphics" = "4.4.2",
  "grDevices" = "4.4.2",
  "Matrix" = "1.7.1",
  "methods" = "4.4.2",
  "stats" = "4.4.2",
  "utils" = "4.4.2",
  "viridisLite" = "0.4.2",
  "optparse" = "1.7.5",
  "dplyr" = "1.1.4",
  "magrittr" = "2.0.3",
  "igraph" = "2.1.4",
  "STRINGdb" = "2.18.0",
  "fields" = "16.3.1",
  "ggplot2" = "3.5.1",
  "writexl" = "1.5.2",
  "biomaRt" = "2.62.1",
  "pdftools" = "3.5.0"
)

if (!requireNamespace("devtools", quietly = TRUE)) install.packages("devtools")

lapply(names(packages), function(pkg) {
  devtools::install_version(pkg, version = packages[pkg])
})
# This script serves as module to perform bioinformatic characterizations for protein list provided by ML workflow
# Example Command line: 
# Rscript Module_ProteinNetwork_WDL.R \
#     --inputPath "/home/rstudio/YD/ML_workflow/input/" \
#     --outPath "/home/rstudio/YD/ML_workflow/output/" \
#     --score_thresholdHere 400 \
#     --combined_score_thresholdHere 800 \
#     --SHAPthresh 100 \
#     --patternChosen "Case1" \
#     --converProId TRUE \
#     >> "/home/rstudio/YD/ML_workflow/output/Network.log" 2>&1 &

library(graphics)
library(grDevices)
library(Matrix)
library(methods)
library(stats)
library(utils)
library(viridisLite)
library(optparse)
library(dplyr)
library(magrittr)
library(igraph)
library(STRINGdb)
library(fields)
library(ggplot2)
library(writexl)
library(biomaRt)
library(pdftools)
#library(GENIE3)
# library(org.Hs.eg.db)
# library(gridExtra)
# library(grid)
set.seed(42)

# Define the list of options
option_list <- list(
  make_option(c("-i", "--inputPath"), type = "character", default = "/home/rstudio/YD/ML_workflow/input/",
              help = "Path to input files.", metavar = "INPUT_PATH"),
  make_option(c("-o", "--outPath"), type = "character", default = "/home/rstudio/YD/ML_workflow/output/",
              help = "Path to output files.", metavar = "OUTPUT_PATH"),
  make_option(c("-s", "--score_thresholdHere"), type = "integer", default = 400, 
              help = "Score threshold for STRING database.", metavar = "SCORE"),
  make_option(c("-c", "--combined_score_thresholdHere"), type = "integer", default = 800, 
              help = "Combined score threshold to select the nodes to be plotted.", metavar = "SCORE"),
  make_option(c("-a", "--SHAPthresh"), type = "integer", default = 100, 
              help = "Shap threshhold to define the top important proteins.", metavar = "ShapThresh"),
  make_option(c("-n", "--patternChosen"), type = "character", default = "", 
              help = "File name pattern defining which SHAP files to be included for analysis.", metavar = "FilePattern"),
  make_option(c("-v", "--converProId"), type = "logical", default = TRUE, 
              help = "Whether to perform the protein name mappin.", metavar = "converProId")
  #make_option(c("-x","--proteinExpFile"), type = "character", default = "label_diagnosis_HCRBD2.csv", 
  #             help = "Protein expression input file name.", metavar = "EXPRESSION")
)

# Create an OptionParser object
parser <- OptionParser(option_list = option_list)

# Parse the command-line arguments
args <- parse_args(parser)

# Assign parsed values to variables
inputPath <- args$inputPath
outPath <- args$outPath
score_thresholdHere <- args$score_thresholdHere
combined_score_thresholdHere <- args$combined_score_thresholdHere
SHAPthresh <- args$SHAPthresh
patternChosen <- args$patternChosen 
converProId = args$converProId
#proteinExpFile <- args$proteinExpFile

#--------------------------------
#--------------------------------
# FUNCTIONS TO BE CALLED BY MAIN
#--------------------------------
#--------------------------------
# Function to generate hub protein table based on centrality score
# Input -- LinkTable: object of links between nodes
# Output -- centrality_df: data frame displaying centrality score for displayed nodes
getHubProTable <- function(LinkTable){
  ### construct igraph object
  igraphMatrixM <- graph_from_data_frame(LinkTable, directed = FALSE) 
  
  ### calculate centrality score
  degreeNODE <- degree(igraphMatrixM, mode="all")
  betweenNODE<- betweenness(igraphMatrixM)
  closeNODE <- closeness(igraphMatrixM)
  
  centrality_df <- data.frame(
    Protein = names(degreeNODE),
    Degree = degreeNODE,
    Betweenness = betweenNODE,
    Closeness = closeNODE
  )
  
  return(centrality_df)
}


# Function to make network plot by mapping important proteins to STRING database
# Input -- Pro_Plot_F: data frame of selected top proteins with corresponding SHAP values, protein name must be Entrez Symbol; 
# Full_SHAP_F_Plot: data frame of all the proteins with corresponding SHAP values, protein name must be Entrez Symbol; 
# score_thresholdHere: score threshold to initialize STRING database;
# combined_score_thresholdHere: score thresold to make network plot;
# patternChosen: message to included in the title of network plot
# Output -- list(Hub_Proteins_STRING, Hub_Proteins_STRING_expanded), protein centrality score for non expanded and expanded protein network.
map2Srting <- function(Pro_Plot_F, Full_SHAP_F_Plot, score_thresholdHere, combined_score_thresholdHere, patternChosen){
  set.seed(42)
  
  ### Initialize STRING database
  string_db <- STRINGdb$new(version = "12", species = 9606, score_threshold = score_thresholdHere, 
                            network_type = "full", input_directory = "", protocol = "http")
  
  mapAll <- string_db$get_aliases()
  
  proName <- Pro_Plot_F$proName
  
  ### Map STRING IDs
  proNameF <- data.frame("prot" = proName)
  stDB_mapped <- string_db$map(proNameF, "prot", removeUnmappedRows = TRUE)
  
  ### Retrieve interactions
  interaction_network <- string_db$get_interactions(stDB_mapped$STRING_id)
  
  ### Filter interactions for high-confidence edges
  edges <- interaction_network %>%
    dplyr::filter(combined_score > combined_score_thresholdHere) %>%
    dplyr::distinct(from, to)
  
  ### Construct igraph object
  g <- graph_from_data_frame(edges, directed = FALSE)
  
  ### Retrieve node names as protein symbols
  nodeName <- V(g)$name
  nodeName2 <- mapAll$alias[unlist(sapply(nodeName, function(x) {
    which(mapAll$STRING_id == x)[1]
  }))]
  
  ### Set node color based on SHAP values
  # Color palette and node coloring
  myPalette <- colorRampPalette(c("darkgreen", "lightgreen", "white", "pink", "darkred"))
  nodeValue <- Pro_Plot_F[nodeName2,"SHAP"]
  nodeColor <- myPalette(1000)[
    as.numeric(cut(nodeValue, breaks = 1000))
  ]
  
  ### Plot the network
  layout_pos <- layout_with_fr(g)
  
  plot(g, 
       layout = layout_pos,
       vertex.label = nodeName2, 
       vertex.size = 3, 
       vertex.label.cex = 0.5, 
       vertex.color = nodeColor, 
       edge.color = "grey", 
       vertex.label.color = "black",
       vertex.label.family = "sans", 
       vertex.label.dist = 0.8, 
       cex.main = 0.06, 
       main = paste0("Protein-Protein Interaction Network ", patternChosen, 
                     "\nBased on STRING database"), 
       rescale = TRUE)
  
  # Add a color legend
  image.plot(legend.only = TRUE, 
             #zlim = range(as.numeric(nodeValue[which(!is.na(nodeValue))])), 
             zlim = c(0,1),
             col = myPalette(1000),
             horizontal = TRUE, 
             legend.width = 0.3, 
             legend.shrink = 0.3, 
             legend.mar = 3,
             legend.args = list(text = "Feature Importance Score", side = 1, font = 2, line = 1, cex = 0.8),
             axis.args = list(cex.axis = 0.8, mgp = c(3, 0.3, 0)))
  
  ### Generate table of centrality score to display hub proteins
  HubProteins <- getHubProTable(edges)
  Hub_Proteins_STRING <- merge(HubProteins, stDB_mapped, by.x = "Protein", by.y = "STRING_id")
  Hub_Proteins_STRING %<>% mutate("Protein" = prot) %<>% dplyr::select(Protein,Betweenness,Closeness,Degree)
  
  ### Further make network plot including more nodes restored in STRINGdb with 1 degree connection with the selected top proteins with highest SHAP values
  # Identify 1st-degree connections
  one_degree_nodes <- unique(c(interaction_network$from, interaction_network$to))
  
  # # Expand mapped list with 1-degree connections
  # expanded_proteins <- mapAll$alias[mapAll$STRING_id %in% one_degree_nodes]
  # 
  # ### Only keep the 1-degree connections in user's own dataset
  # expanded_proteins_keep  <- expanded_proteins[which(expanded_proteins %in% rownames(Full_SHAP_F_Plot))]
  
  expanded_proteins <- mapAll$STRING_id[mapAll$STRING_id %in% one_degree_nodes]
  
  ensembl <- useMart("ensembl", dataset = "hsapiens_gene_ensembl")
  AllUserPro <- getBM(
    attributes = c("hgnc_symbol", "ensembl_gene_id"),  # Mapping HGNC symbol to Ensembl ID
    filters = "hgnc_symbol",
    values = rownames(Full_SHAP_F_Plot),
    mart = ensembl
  )
  
  expanded_proteins_keep  <- expanded_proteins[which(gsub("9606.", "", expanded_proteins) %in% AllUserPro$ensembl_gene_id)]
  
  # Extract nodes without extension
  current_nodes <- unique(c(interaction_network$from, interaction_network$to))
  
  # Identify additional proteins with direct connections
  additional_mapped <- string_db$map(data.frame("prot" = expanded_proteins_keep), "prot", removeUnmappedRows = TRUE)
  
  # Retrieve interactions involving additional proteins
  additional_interactions <- string_db$get_interactions(additional_mapped$STRING_id)
  
  # Filter to keep only interactions with existing nodes
  filtered_additional_interactions <- additional_interactions %>%
    dplyr::filter(from %in% current_nodes | to %in% current_nodes)
  
  # Combine both interaction datasets and remove duplicates
  expanded_interaction_network <- bind_rows(interaction_network, filtered_additional_interactions) %>%
    distinct(from, to, .keep_all = TRUE)
  
  ### Filter interactions for high-confidence edges
  edges_expanded <- expanded_interaction_network %>%
    dplyr::filter(combined_score > combined_score_thresholdHere) %>%
    dplyr::distinct(from, to)
  
  ### Construct igraph object
  g_expanded <- graph_from_data_frame(edges_expanded, directed = FALSE)
  
  ### Retrieve node names as protein symbols
  nodeName_expanded <- V(g_expanded)$name
  nodeName2_expanded <- mapAll$alias[unlist(sapply(nodeName_expanded, function(x) {
    which(mapAll$STRING_id == x)[1]
  }))]
  
  nodeValue_expanded <- Full_SHAP_F_Plot[nodeName2, "SHAP"]
  
  # Color palette and node coloring
  nodeColor_expanded <- myPalette(1000)[
    as.numeric(cut(nodeValue_expanded, breaks = 1000))
  ]
  
  ### Plot the network
  layout_pos_expanded <- layout_with_fr(g_expanded)
  
  plot(g_expanded, 
       layout = layout_pos_expanded, 
       vertex.label = nodeName2_expanded, 
       vertex.size = 3, 
       vertex.label.cex = 0.5, 
       vertex.color = nodeColor_expanded, 
       edge.color = "grey", 
       vertex.label.color = "black",
       vertex.label.family = "sans", 
       vertex.label.dist = 0.8, 
       cex.main = 0.06, 
       main = paste0("Protein-Protein expanded Interaction Network ", patternChosen, 
                     "\nBased on STRING database"), 
       rescale = TRUE)
  
  # Add a color legend
  image.plot(legend.only = TRUE, 
             #zlim = range(as.numeric(nodeValue_expanded[which(!is.na(nodeValue_expanded))])), 
             zlim = c(0,1),
             col = myPalette(1000),
             horizontal = TRUE, 
             legend.width = 0.3, 
             legend.shrink = 0.3, 
             legend.mar = 3,
             legend.args = list(text = "Feature Importance Score", side = 1, font = 2, line = 1, cex = 0.8),
             axis.args = list(cex.axis = 0.8, mgp = c(3, 0.3, 0)))
  
  ### Generate table of centrality score to display hub proteins
  HubProteins_expanded <- getHubProTable(edges_expanded)
  Hub_Proteins_STRING_expanded <- merge(HubProteins_expanded, stDB_mapped, by.x = "Protein", by.y = "STRING_id")
  Hub_Proteins_STRING_expanded %<>% mutate("Protein" = prot) %<>% dplyr::select(Protein,Betweenness,Closeness,Degree)
  
  return(list(Hub_Proteins_STRING, Hub_Proteins_STRING_expanded))
}

#--------
#--------
# MAIN
#--------
#--------
################
# READ IN DATA
################
print(paste0("Network plot for ",  patternChosen, " started at ", format(Sys.time(), "%H:%M:%S"), " on ", Sys.Date(),"."))

# Choose which SHAP files and which SHAP threshold to make network plot
if(patternChosen==""){proteinImportanceFile_list <- list.files(inputPath)
}else{proteinImportanceFile_list <- list.files(inputPath, pattern=patternChosen)}

ModelName <- gsub("_shap_values.csv", "",proteinImportanceFile_list)
ProImportance <- vector("list", length=length(proteinImportanceFile_list))
names(ProImportance) <- ModelName

for(fCt in 1:length(proteinImportanceFile_list)){
  ProImportance_temp <- read.csv(paste0(inputPath, proteinImportanceFile_list[fCt]), row.names=1)
  ProImportance_temp %<>% mutate(SHAP = rowMeans(across(everything(), abs))) %<>% arrange(desc(SHAP)) %<>% dplyr::select("SHAP") 
  ProImportance[[fCt]] <- ProImportance_temp
}

ProImportance_aligned <- lapply(ProImportance, function(df) {
  df <- as.data.frame(df)
  df$Protein <- rownames(df)  # Store row names in a column for merging
  return(df)
})

### Data frame with all the proteins and corresponding SHAP values without filtering
NewSHAP <- Reduce(function(x, y) full_join(x, y, by = "Protein"), ProImportance_aligned)

# Set row names back to "Protein" and remove the extra column. This is indispensable for SHAP values derived from multiple importance files
rownames(NewSHAP) <- NewSHAP$Protein
NewSHAP$Protein <- NULL
colnames(NewSHAP) <- ModelName

### Keep NewSHAP for further reference
NewSHAP_scaled <- as.data.frame(lapply(NewSHAP, function(x) {
  (x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE))
})) ### standardize the magnitude per model to 0~1
rownames(NewSHAP_scaled) <- rownames(NewSHAP)

CombinedShap <- NewSHAP_scaled %>% mutate(CombinedShap = rowMeans(across(everything(), abs), na.rm = TRUE)) %>% dplyr::select("CombinedShap")
Full_SHAP_F <- cbind(NewSHAP,CombinedShap) %>% arrange(desc(CombinedShap))
Full_SHAP_Ori <- rownames(Full_SHAP_F)
Full_SHAP_F_AllScaled <- cbind(NewSHAP_scaled,CombinedShap) %>% arrange(desc(CombinedShap))

### Output network plots and hub protein tables
pdf_fileName <- paste0(colCt, "Network.pdf")
pdf_filePath <- paste0(outPath, pdf_fileName)
pdf(pdf_filePath)

for(colCt in colnames(Full_SHAP_F_AllScaled)){
  SHAP_PlotF <- Full_SHAP_F_AllScaled %>% dplyr::select(all_of(colCt)) %>% arrange(desc(!!sym(colCt))) %>%slice_head(n = SHAPthresh) 
  
  ### Prepare data frame of top proteins with highest importance
  Pro_Plot_Ori <- rownames(SHAP_PlotF)
  
  ### Entrez Symbol is used to display on the network plot, perform the protein name mapping
  if(converProId == TRUE){
    ensembl <- useMart("ensembl", dataset = "hsapiens_gene_ensembl")
    Pro_Plot <- getBM(
      attributes = c("uniprotswissprot", "hgnc_symbol"),
      filters = "uniprotswissprot",
      values = Pro_Plot_Ori,
      mart = ensembl
    )
    SHAP <- unlist(sapply(Pro_Plot$uniprotswissprot, function(x) {SHAP_PlotF[which(rownames(SHAP_PlotF)==x)[1], colCt]}))
    Pro_Plot_F <- Pro_Plot %>% dplyr::select(hgnc_symbol) %>% mutate(SHAP = SHAP) %>% arrange(desc(SHAP))
    
    # Wait for a specified time (e.g., 10 seconds to avoid server Ensembl’s biomaRt API crush)
    Sys.sleep(10)  
    
    ### We also need to retreive all the SHAP values for all the proteins for the purpose of expansion network plot
    Full_SHAP <- getBM(
      attributes = c("uniprotswissprot", "hgnc_symbol"),
      filters = "uniprotswissprot",
      values = Full_SHAP_Ori,
      mart = ensembl
    )
    SHAP_All <- unlist(sapply(Full_SHAP$uniprotswissprot, function(x) {Full_SHAP_F[which(rownames(Full_SHAP_F)==x)[1], colCt]}))
    Full_SHAP_F_Plot <- cbind(Full_SHAP$hgnc_symbol, SHAP_All) %>% as.data.frame() %>% mutate(SHAP_All = as.numeric(SHAP_All))
    
  }else{
    Pro_Plot_F <- cbind(rownames(SHAP_PlotF), SHAP_PlotF[,colCt]) %>% as.data.frame()
    Full_SHAP_F_Plot <- cbind(rownames(Full_SHAP_F),Full_SHAP_F[,colCt]) %>% as.data.frame()
  }
  
  ###  Set the arbitrary colnames for the convenience of processing data.
  colnames(Pro_Plot_F) <- colnames(Full_SHAP_F_Plot) <- c("proName", "SHAP")
  rownames(Pro_Plot_F) <- Pro_Plot_F$proName
  
  ### Tackle with the same UniProtID corresponding to different Entrez Symbols
  Full_SHAP_F_Plot %<>% group_by(proName) %<>% summarise(SHAP = max(SHAP, na.rm = TRUE), .groups = "drop") %<>% as.data.frame()
  rownames(Full_SHAP_F_Plot) <- Full_SHAP_F_Plot$proName
  
  ###############
  # OUTPUT FILES
  ###############
  Hub_Proteins_STRING_List <- map2Srting(Pro_Plot_F, Full_SHAP_F_Plot, score_thresholdHere, combined_score_thresholdHere, colCt)
  Hub_Proteins_STRING <- Hub_Proteins_STRING_List[[1]] %>% arrange(desc(Degree))
  Hub_Proteins_STRING_WithExpansion <- Hub_Proteins_STRING_List[[2]] %>% arrange(desc(Degree))
  
  write_xlsx(Hub_Proteins_STRING, path = paste0(outPath, colCt, "_Hub_Proteins_STRING.xlsx"))
  write_xlsx(Hub_Proteins_STRING_WithExpansion, path = paste0(outPath, colCt, "_Hub_Proteins_STRING_WithExpansion.xlsx"))
}

dev.off()

### Generate png for the convenience to combine into the final pdf
num_pages <- pdf_info(pdf_filePath)$pages
for (i in 1:num_pages) {
  pdf_subset(paste0(outPath, pdf_fileName), pages = i, output = paste0(outPath, "Network_", i, ".pdf"))
  
  pdf_convert(paste0(outPath, "Network_", i, ".pdf"), format = "png", filenames = paste0(outPath, "Network_", i, ".png"), dpi = 300)
}

print(paste0("Network plot for ",  colCt, " finished at ", format(Sys.time(), "%H:%M:%S"), " on ", Sys.Date(),"."))

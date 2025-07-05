# This script is a module for performing protein-protein interaction (PPI) network analysis on proteins selected based on protein importance derived from classification or regression machine learning models.

# Example Command line for case: 
# Rscript protein_network.R \
#     --score_thresholdHere 400 \
#     --combined_score_thresholdHere 800 \
#     --SHAPthresh 100 \
#     --patternChosen "shap_values.csv" \
#     --converProId TRUE \
#     --proteinExpFile "Case1-1.csv" \
#     --CorMethod "spearman" \
#     --CorThreshold 0.8 \
#     > "/home/rstudio/YD/ML_workflow_DY/output/Network_WT.log" 2>&1 

library(optparse)
library(tidyverse)
library(magrittr)
library(igraph)
library(STRINGdb)
library(fields)
library(ggplot2)
library(org.Hs.eg.db)
library(AnnotationDbi)
library(writexl)
library(pdftools)

set.seed(42)
suppressWarnings(library(AnnotationDbi))
suppressMessages(library(AnnotationDbi))

### Define the list of options
option_list <- list(
  make_option(c("-s", "--score_thresholdHere"), type = "integer", default = 400, 
              help = "Confidence score threshold for loading STRING database.", metavar = "SCORE"),
  make_option(c("-c", "--combined_score_thresholdHere"), type = "integer", default = 800, 
              help = "Confidence score threshold for selecting nodes to plot in the network.", metavar = "SCORE"),
  make_option(c("-a", "--SHAPthresh"), type = "integer", default = 100, 
              help = "The number of top important proteins used for network analysis.", metavar = "ShapThresh"),
  make_option(c("-n", "--patternChosen"), type = "character", default = "shap_values.csv", 
              help = "File name pattern defining which SHAP files to be included for analysis.", metavar = "FilePattern"),
  make_option(c("-v", "--converProId"), type = "logical", default = TRUE, 
              help = "Whether to perform protein name mapping from UniProt IDs to Entrez Gene Symbols.", metavar = "converProId"),
  make_option(c("-x","--proteinExpFile"), type = "character", default = "Case1-1.csv", 
              help = "Name of the input file containing the protein expression profile.", metavar = "EXPRESSION"),
  make_option(c("-m","--CorMethod"), type = "character", default = "spearman", 
              help = "Correlation method used to define strongly co-expressed proteins; choose from Spearman, Pearson, or Kendall.", metavar = "CorMethod"),
  make_option(c("-d","--CorThreshold"), type = "numeric", default = 0.8, 
              help = "Threshold value of the correlation coefficient used to identify strongly co-expressed proteins.", metavar = "CorThreshold")
)

### Create an OptionParser object
parser <- OptionParser(option_list = option_list)

### Parse the command-line arguments
args <- parse_args(parser)

### Assign parsed values to variables
score_thresholdHere <- args$score_thresholdHere
combined_score_thresholdHere <- args$combined_score_thresholdHere
SHAPthresh <- args$SHAPthresh
patternChosen <- args$patternChosen 
converProId = args$converProId
proteinExpFile <- args$proteinExpFile
CorMethod <- args$CorMethod
CorThreshold <- args$CorThreshold

#--------------------------------
#--------------------------------
# FUNCTIONS TO BE CALLED BY MAIN
#--------------------------------
#--------------------------------
# Function to map UniProt IDs to Entrez Gene Symbols
# Input: UniProList – a vector of UniProt IDs
# Output: DataF_EntrezSym – a data frame mapping each UniProt ID to its corresponding Entrez Gene Symbol
ConvertUniprot2Symbol <- function(UniProList){
  DataF_EntrezSym <- AnnotationDbi::select(
    org.Hs.eg.db,
    keys = UniProList,
    keytype = "UNIPROT",
    columns = c("UNIPROT", "SYMBOL")
  ) %>%
    dplyr::group_by(UNIPROT) %>%
    dplyr::summarize(SYMBOL = paste(na.omit(unique(SYMBOL)), collapse = ";"), .groups = "drop") %>% ### UniProt IDs with duplicated or multiple associated gene symbols
    dplyr::filter(!is.na(SYMBOL) & SYMBOL != "" & SYMBOL != "NA" & SYMBOL != " ") %>% as.data.frame() %>% ### Possible entries with no mapped gene symbol for the given UniProt ID in the database 
    distinct(SYMBOL, .keep_all = TRUE) ### Multiple UniProt IDs mapping to the same Entrez symbol: Only the first occurrence is retained in the final dataset.
  
  return(DataF_EntrezSym)
}

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
# Input -- Pro_Plot_F: SHAP-ordered data frame of selected top proteins with corresponding SHAP values, protein name must be Entrez Symbol; 
# Full_SHAP_F_Plot: data frame of all the proteins with corresponding SHAP values, protein name must be Entrez Symbol; 
# score_thresholdHere: score threshold to initialize STRING database;
# combined_score_thresholdHere: score thresold to select nodes and make network plot;
# CoPro_EntrezSym: Entrez Symbols of proteins strongly co-expressed with the top important proteins identified by SHAP values;
# patternChosen: message to included in the title of network plot;
# Output -- Plots of networks composed of top important proteins identified by SHAP values;
#           list(Hub_Proteins_STRING, Hub_Proteins_STRING_expanded), protein centrality score for non expanded and expanded protein network.
map2String <- function(string_db, Pro_Plot_F, Full_SHAP_F_Plot, score_thresholdHere, combined_score_thresholdHere, CoPro_EntrezSym, patternChosen){
  set.seed(42)
  
  mapAll <- string_db$get_aliases()
  
  proName <- Pro_Plot_F$proName
  
  ### Map STRING IDs
  proNameF <- data.frame("prot" = proName)
  stDB_mapped <- string_db$map(proNameF, "prot", removeUnmappedRows = TRUE)
  
  ### Retrieve interactions
  interaction_network <- string_db$get_interactions(stDB_mapped$STRING_id) %>%
    distinct(from, to, .keep_all = TRUE)
  
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
  ### Color palette and node coloring
  myPalette <- colorRampPalette(c("darkgreen", "lightgreen", "white", "pink", "darkred"))
  nodeValue <- as.numeric(Pro_Plot_F[nodeName2,"SHAP"])
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
       cex.main = 0.03, 
       main = paste0("Protein-Protein Interaction Network\n", patternChosen), 
       rescale = TRUE)
  
  ### Add a color legend
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
  ### Identify all the interactions between proteins in our full protein list
  all_mapped <- string_db$map(data.frame("prot" = Full_SHAP_F_Plot$proName), "prot", removeUnmappedRows = TRUE)
  
  ### Retrieve interactions involving all proteins
  all_interaction_network <- string_db$get_interactions(all_mapped$STRING_id) %>%
    distinct(from, to, .keep_all = TRUE)
  
  ### Identify 1st-degree seed nodes (top important proteins)
  one_degree_nodes <- unique(c(interaction_network$from, interaction_network$to))
  
  ### Identify highly coexpressed proteins with the seed nodes
  CoPro_mapped <- string_db$map(data.frame("prot" = CoPro_EntrezSym), "prot", removeUnmappedRows = TRUE)
  CoPro_ENSEMBL <- CoPro_mapped$STRING_id
  
  ### Within the all_interaction_network, identify which interactions include the seed node or 1st degree connection between seed node and strongly coexpressed proteins 
  keepID <- which(
    (all_interaction_network$from %in% one_degree_nodes & 
       all_interaction_network$to %in% CoPro_ENSEMBL) |
      
      (all_interaction_network$from %in% CoPro_ENSEMBL & 
         all_interaction_network$to %in% one_degree_nodes) |
      
      (all_interaction_network$from %in% one_degree_nodes & 
         all_interaction_network$to %in% one_degree_nodes)
  )
  
  ### Include the original network only composed of the top important proteins based on SHAP
  expanded_interaction_network <- rbind(all_interaction_network[keepID, ],interaction_network) %>%
    distinct(from, to, .keep_all = TRUE)
  
  ### Filter interactions for high-confidence edges
  edges_expanded <- expanded_interaction_network %>%
    dplyr::filter(combined_score > combined_score_thresholdHere) %>%
    dplyr::distinct(from, to, .keep_all = TRUE)
  
  ### Construct igraph object
  g_expanded <- graph_from_data_frame(edges_expanded, directed = FALSE)
  
  ### Retrieve node names as protein symbols
  nodeName_expanded <- V(g_expanded)$name
  nodeName2_expanded <- mapAll$alias[unlist(sapply(nodeName_expanded, function(x) {
    which(mapAll$STRING_id == x)[1]
  }))]
  
  nodeValue_expanded <- as.numeric(Full_SHAP_F_Plot[nodeName2_expanded, "SHAP"])
  
  ### Color palette and node coloring
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
       cex.main = 0.03, 
       main = paste0("Protein-Protein Expanded Interaction Network\n", patternChosen), 
       rescale = TRUE)
  
  ### Add a color legend
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
  Hub_Proteins_STRING_expanded <- merge(HubProteins_expanded, all_mapped, by.x = "Protein", by.y = "STRING_id")
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
print(paste0("PPI network and Hub Protein analysis based on ",  patternChosen, " started at ", format(Sys.time(), "%H:%M:%S"), " on ", Sys.Date(),"."))

### Choose which SHAP files and which SHAP threshold to make network plot
proteinImportanceFile_list <- list.files("./" ,pattern=patternChosen)

ModelName <- gsub("_shap_values.csv", "",proteinImportanceFile_list)
ProImportance <- vector("list", length=length(proteinImportanceFile_list))
names(ProImportance) <- ModelName

for(fCt in 1:length(proteinImportanceFile_list)){
  ProImportance_temp <- read.csv(paste0(proteinImportanceFile_list[fCt]), row.names=1)
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

### Set row names back to "Protein" and remove the extra column. This is indispensable for SHAP values derived from multiple importance files
rownames(NewSHAP) <- NewSHAP$Protein
NewSHAP$Protein <- NULL
colnames(NewSHAP) <- ModelName

### Keep NewSHAP for further reference
NewSHAP_scaled <- as.data.frame(lapply(NewSHAP, function(x) {
  if(max(x, na.rm = TRUE) - min(x, na.rm = TRUE) == 0){
    rep(0, length(x)) # if all the proteins have the same SHAP values
  }else{(x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE))}
})) ### standardize the magnitude per model to 0~1
colnames(NewSHAP_scaled) <- colnames(NewSHAP)
rownames(NewSHAP_scaled) <- rownames(NewSHAP)

CombinedShap <- NewSHAP_scaled %>% mutate(CombinedShap = rowMeans(across(everything(), abs), na.rm = TRUE)) %>% dplyr::select("CombinedShap")
Full_SHAP_F <- cbind(NewSHAP,CombinedShap) %>% arrange(desc(CombinedShap))
Full_SHAP_Ori <- rownames(Full_SHAP_F)
Full_SHAP_F_AllScaled <- cbind(NewSHAP_scaled,CombinedShap) %>% arrange(desc(CombinedShap))

### Output network plots and hub protein tables
pdf_fileName <- "Network.pdf"
pdf(pdf_fileName)

### Read in protein expression profile 
proExpF <- read.csv(proteinExpFile, check.names=FALSE)[,c(-1,-2)] %>% dplyr::select(where(~ any(. != 0))) ### filter out proteins with all-zero values

for(colCt in colnames(Full_SHAP_F_AllScaled)[!grepl("CombinedShap", colnames(Full_SHAP_F_AllScaled))]){
  print(paste0("Proteins with the highest importance scores based on ", colCt, " are selected for PPI analysis."))
  
  tryCatch(
    {if(all(Full_SHAP_F_AllScaled[,colCt] == 0)){message(paste0("For ", colCt, " All proteins have identical importance scores, so the most important ones cannot be distinguished."))
    }else{
      SHAP_PlotF <- Full_SHAP_F_AllScaled %>% dplyr::select(all_of(colCt)) %>% arrange(desc(!!sym(colCt))) %>% filter(!!sym(colCt) != 0) 
      
      # Handle case where the requested number of proteins exceeds available non-zero SHAP values
      if (SHAPthresh > nrow(SHAP_PlotF)) {
        message(paste0("For ", colCt, " Not enough proteins with non-zero SHAP values for the requested threshold of ", SHAPthresh, ". Using all available proteins with non-zero SHAP values for PPI analysis instead."
        ))
        SHAPthresh <- nrow(SHAP_PlotF)
      }
      
      SHAP_PlotF %<>% slice_head(n = SHAPthresh) 
      
      ### Prepare data frame of top proteins with highest importance
      Pro_Plot_Ori <- intersect(rownames(SHAP_PlotF), colnames(proExpF))
      
      ### Identify the strongly coexpressed proteins with the top important proteins based on SHAP
      corF <- cor(proExpF,method = CorMethod)[Pro_Plot_Ori,]
      CoPro_UniPro <- unique(colnames(corF)[which(abs(corF) > CorThreshold, arr.ind = TRUE)[, 2]])
      
      ### Entrez Symbol is used to display on the network plot, perform the protein name mapping
      if(converProId == TRUE){
        
        ### Retrieve SHAP values for all proteins to generate the expansion network plot.
        ### Map UniProt IDs to Entrez symbols for all proteins. NAs might be introduced due to missing mapping information in the database.
        Full_SHAP <- ConvertUniprot2Symbol(Full_SHAP_Ori)
        SHAP_All <- unlist(sapply(Full_SHAP$UNIPROT, function(x) {Full_SHAP_F[which(rownames(Full_SHAP_F)==x)[1], colCt]}))
        Full_SHAP_F_Plot <- cbind(Full_SHAP$SYMBOL, SHAP_All) %>% as.data.frame() %>% mutate(SHAP_All = as.numeric(SHAP_All))
        colnames(Full_SHAP_F_Plot) <- c("SYMBOL", "SHAP")
        
        ### SHAP frame for the top important proteins
        Pro_Plot_F <- Full_SHAP_F_Plot[intersect(Pro_Plot_Ori, rownames(Full_SHAP_F_Plot)),] %>% arrange(desc(SHAP))
        
        ### Map strongly coexpressed proteins between UniProt Id and Entrez Symbol
        CoPro_EntrezSym <- Full_SHAP_F_Plot[intersect(CoPro_UniPro, rownames(Full_SHAP_F_Plot)), "SYMBOL"]
        
      }else{
        Pro_Plot_F <- cbind(rownames(SHAP_PlotF), SHAP_PlotF[,colCt]) %>% as.data.frame()
        Full_SHAP_F_Plot <- cbind(rownames(Full_SHAP_F),Full_SHAP_F[,colCt]) %>% as.data.frame()
        CoPro_EntrezSym <- CoPro_UniPro
      }
      
      ### Set the arbitrary colnames for the convenience of processing data.
      colnames(Pro_Plot_F) <- colnames(Full_SHAP_F_Plot) <- c("proName", "SHAP")
      rownames(Pro_Plot_F) <- Pro_Plot_F$proName
      
      ### Tackle with the same UniProtID corresponding to different Entrez Symbols and remove the NA SHAP values.
      Full_SHAP_F_Plot %<>% filter(!(is.na(SHAP)))%<>% group_by(proName) %<>% summarise(SHAP = max(SHAP, na.rm = TRUE), .groups = "drop") %<>% as.data.frame()
      rownames(Full_SHAP_F_Plot) <- Full_SHAP_F_Plot$proName
      
      ### Pro_Plot_F is the SHAP-ordered frame only for the top important proteins; Full_SHAP_F_Plot is NOT SHAP-ordered though for all the proteins.
      
      ###############
      # OUTPUT FILES
      ###############
      ### Initialise the STRINGdb database
      string_db <- STRINGdb$new(version = "12", species = 9606, score_threshold = score_thresholdHere, 
                                network_type = "full", input_directory = "/scripts", protocol = "http")
      
      Hub_Proteins_STRING_List <- map2String(string_db, Pro_Plot_F, Full_SHAP_F_Plot, score_thresholdHere, combined_score_thresholdHere, CoPro_EntrezSym, colCt)
      Hub_Proteins_STRING <- Hub_Proteins_STRING_List[[1]] %>% arrange(desc(Degree))
      Hub_Proteins_STRING_WithExpansion <- Hub_Proteins_STRING_List[[2]] %>% arrange(desc(Degree))
      
      write_xlsx(Hub_Proteins_STRING, path = paste0(colCt, "_Hub_Proteins_STRING.xlsx"))
      write_xlsx(Hub_Proteins_STRING_WithExpansion, path = paste0(colCt, "_Hub_Proteins_STRING_WithExpansion.xlsx"))
    }
      error = function(e) {
        message(paste0("For ", colCt, ", The PPI network plot could not be generated. Please ensure that (1) protein names are uniquely provided as either Entrez Gene Symbols or UniProt IDs, and (2) the number of top important proteins is properly selected with sufficient number of non-NA SHAP values present."))
      }
    }
  )
}

dev.off()

### Generate png for the convenience to combine into the final pdf
num_pages <- pdf_info(pdf_fileName)$pages
for (i in 1:num_pages) {
  pdf_subset(pdf_fileName, pages = i, output = paste0("Network_", i, ".pdf"))
  
  pdf_convert(paste0("Network_", i, ".pdf"), format = "png", filenames = paste0("Network_", i, ".png"), dpi = 300)
}

print(paste0("PPI network and Hub Protein analysis finished at ", format(Sys.time(), "%H:%M:%S"), " on ", Sys.Date(),"."))
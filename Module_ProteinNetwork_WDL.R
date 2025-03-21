# This script serves as module to perform bioinformatic characterizations for protein list provided by ML workflow

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
library(GENIE3)
library(igraph)
library(STRINGdb)
library(fields)
library(ggplot2)
library(biomaRt)
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
  make_option(c("-t", "--titleMessage"), type = "character", default = "Case1", 
              help = "Input title message for the output plots.", metavar = "MESSAGE"),
  make_option(c("-s", "--score_thresholdHere"), type = "integer", default = 400, 
              help = "Score threshold for STRING database.", metavar = "SCORE"),
  make_option(c("-c", "--combined_score_thresholdHere"), type = "integer", default = 500, 
              help = "Combined score threshold to select the nodes to be plotted.", metavar = "SCORE"),
  make_option(c("-a", "--SHAPthresh"), type = "integer", default = 100, 
              help = "Shap threshhold to define the top important proteins.", metavar = "ShapThresh"),
  make_option(c("-n", "--patternChosen"), type = "character", default = "Case1_lightgbm", 
              help = "File name pattern defining which SHAP files to be included for analysis.", metavar = "FilePattern"),
  make_option(c("-v", "--converProId"), type = "logical", default = TRUE, 
              help = "Whether to perform the protein name mappin.", metavar = "converProId"),
  make_option(c("-R", "--RFedge_threshold_here"), type = "numeric", default = 0.04, 
              help = "RF edge threshold to be plotted.", metavar = "THRESHOLD"),
  make_option(c("-K", "--KnodeHere"), type = "integer", default = 10, 
              help = "Number of random nodes selected per iteration for RF.", metavar = "NODES"),
  make_option(c("-N", "--nTreeHere"), type = "integer", default = 10, 
              help = "Number of trees built per iteration for RF.", metavar = "TREE"),
  make_option(c("-p","--proteinImportanceFile"), type = "character", default = "/home/rstudio/YD/ML_workflow/input/Case1_lightgbm_shap_values.csv", 
              help = "Protein importance input file name.", metavar = "IMPORTANCE")
  # make_option(c("-x","--proteinExpFile"), type = "character", default = "label_diagnosis_HCRBD2.csv", 
  #             help = "Protein expression input file name.", metavar = "EXPRESSION")
)

# Create an OptionParser object
parser <- OptionParser(option_list = option_list)

# Parse the command-line arguments
args <- parse_args(parser)

# Assign parsed values to variables
inputPath <- args$inputPath
outPath <- args$outPath
proteinImportanceFile <- args$proteinImportanceFile
score_thresholdHere <- args$score_thresholdHere
combined_score_thresholdHere <- args$combined_score_thresholdHere
SHAPthresh <- args$SHAPthresh
patternChosen <- args$patternChosen 
converProId = args$converProId
titleMessage <- args$titleMessage
# RFedge_threshold_here <- args$RFedge_threshold_here
# KnodeHere <- args$KnodeHere
# nTreeHere <- args$nTreeHere
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
# titleMessage: message to included in the title of network plot
# Output -- list(Hub_Proteins_STRING, Hub_Proteins_STRING_extended), protein centrality score for non extended and extended protein network.
map2Srting <- function(Pro_Plot_F, Full_SHAP_F_Plot, score_thresholdHere, combined_score_thresholdHere, titleMessage){
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
  nodeValue <- Pro_Plot_F[nodeName2,"SHAP"]
  nodeColor <- myPalette(1000)[
    as.numeric(cut(nodeValue, breaks = 1000))
  ]
  
  # Color palette and node coloring
  myPalette <- colorRampPalette(c("darkgreen", "lightgreen", "white", "pink", "darkred"))
  
  ### Plot the network
  plot(g, 
       vertex.label = nodeName2, 
       vertex.size = 3, 
       vertex.label.cex = 0.5, 
       vertex.color = nodeColor, 
       edge.color = "grey", 
       vertex.label.color = "black",
       vertex.label.family = "sans", 
       vertex.label.dist = 0.8, 
       cex.main = 0.06, 
       main = paste0("Protein-Protein Interaction Network", titleMessage, 
                     "\nBased on STRING database"), 
       rescale = TRUE)
  
  # Add a color legend
  image.plot(legend.only = TRUE, 
             zlim = range(as.numeric(nodeValue[which(!is.na(nodeValue))])), 
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
  
  # Expand mapped list with 1-degree connections
  expanded_proteins <- mapAll$alias[mapAll$STRING_id %in% one_degree_nodes]
  
  ### Only keep the 1-degree connections in user's own dataset
  expanded_proteins_keep  <- expanded_proteins[which(expanded_proteins %in% rownames(Full_SHAP_F_Plot))]
  
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
  
  nodeValue_expanded <- NewSHAP[nodeName2, "CombinedShap"]
  
  # Color palette and node coloring
  nodeColor_expanded <- myPalette(1000)[
    as.numeric(cut(nodeValue_expanded, breaks = 1000))
  ]
  
  ### Plot the network
  plot(g_expanded, 
       vertex.label = nodeName2_expanded, 
       vertex.size = 3, 
       vertex.label.cex = 0.5, 
       vertex.color = nodeColor_expanded, 
       edge.color = "grey", 
       vertex.label.color = "black",
       vertex.label.family = "sans", 
       vertex.label.dist = 0.8, 
       cex.main = 0.06, 
       main = paste0("Protein-Protein Interaction Network ", titleMessage, 
                     "\nBased on STRING database"), 
       rescale = TRUE)
  
  # Add a color legend
  image.plot(legend.only = TRUE, 
             zlim = range(as.numeric(nodeValue_expanded[which(!is.na(nodeValue_expanded))])), 
             col = myPalette(1000),
             horizontal = TRUE, 
             legend.width = 0.3, 
             legend.shrink = 0.3, 
             legend.mar = 3,
             legend.args = list(text = "Feature Importance Score", side = 1, font = 2, line = 1, cex = 0.8),
             axis.args = list(cex.axis = 0.8, mgp = c(3, 0.3, 0)))
  
  ### Generate table of centrality score to display hub proteins
  HubProteins_extended <- getHubProTable(edges_expanded)
  Hub_Proteins_STRING_extended <- merge(HubProteins_extended, stDB_mapped, by.x = "Protein", by.y = "STRING_id")
  Hub_Proteins_STRING_extended %<>% mutate("Protein" = prot) %<>% dplyr::select(Protein,Betweenness,Closeness,Degree)
  
  return(list(Hub_Proteins_STRING, Hub_Proteins_STRING_extended))
}

### Make network plot by applying GENEI3 Random Forrest Algorithm bsaed on protein expression data
# regulatoryNet_WF <- function(ProImportanceF,proDFhere,KnodeHere,nTreeHere,RFedge_threshold_here,titleMessage){
#   set.seed(42)
#   ### expression profile of master regulators applied to GENIE3
#   proName <- rownames(ProImportanceF)
#   masterRexp = proDFhere[,intersect(proName,colnames(proDFhere))]
#   
#   ### construct the tree
#   weightMat <- GENIE3(t(masterRexp), treeMethod="ET", K=KnodeHere, nTrees=nTreeHere)
#   netRaw <- getLinkList(weightMat,threshold=RFedge_threshold_here)
#   
#   ### retrieve regulatory network nodes and edge information, plot network
#   adjM <- as_adjacency_matrix(graph_from_edgelist(as.matrix(netRaw[,1:2]), directed=TRUE))
#   gra <- graph_from_adjacency_matrix(adjM, mode = "directed", weighted = TRUE,diag = FALSE, add.colnames = NULL, add.rownames = NA)
#   vertex.display <- V(gra)$name
#   
#   ### set node color
#   nodeValue <- ProImportanceF[vertex.display,]
#   names(nodeValue) <- vertex.display
#   
#   nodeColor <- nodeValueNorm <- sapply(nodeValue,function(x){
#     (x - min(nodeValue)) / (max(nodeValue) - min(nodeValue))
#   })
#   
#   myPalette <- colorRampPalette(c("darkgreen","lightgreen","white","pink","darkred"))
#   nodeColor <- myPalette(1000)[as.numeric(cut(as.numeric(nodeColor), breaks=1000))]
#   names(nodeColor) <- names(nodeValue)
#   
#   plot(gra, main=paste0("Protein Regulatory Network\n",titleMessage), cex.main=0.8,
#        vertex.label=vertex.display,vertex.size=3, 
#        vertex.color=nodeColor[vertex.display],vertex.label.font=0.5, vertex.label.cex=.5, vertex.label.color="darkgreen",
#        edge.arrow.size=0.2, arrow.width=1, vertex.label.dist=1.5,rescale = TRUE)
#   
#   image.plot(legend.only=TRUE, zlim=range(nodeValue), col=myPalette(1000),
#              horizontal = TRUE,legend.width=0.3, legend.shrink=0.3, legend.mar=3, 
#              legend.args=list(text="Feature Importance Score", side=1, font=2, line=1, cex=0.5),
#              axis.args = list(cex.axis = 0.5,mgp = c(3, 0.3, 0)))
#   
#   ### Generate table of centrality score to display hub proteins
#   Hub_Proteins <- getHubProTable(netRaw)
#   
#   return(Hub_Proteins)
# }

#--------
#--------
# MAIN
#--------
#--------
################
# READ IN DATA
################
# Choose which SHAP files and which SHAP threshold to make network plot
proteinImportanceFile_list <- list.files(inputPath, pattern=patternChosen)
ModelName <- gsub("_shap_values.csv", "",proteinImportanceFile_list)
ProImportance <- BiomarkerList <- ProteinName <- vector("list", length=length(proteinImportanceFile_list))
names(ProImportance) <- names(ProImportance) <- names(BiomarkerList) <- names(ProteinName) <- ModelName

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
Full_SHAP_F <- NewSHAP %>% mutate(CombinedShap = rowMeans(across(everything(), abs))) %>% arrange(desc(CombinedShap))
Full_SHAP_Ori <- rownames(Full_SHAP_F)

### Prepare data frame of top proteins with highest importance
SHAP_PlotF <- Full_SHAP_F %>% slice_head(n = SHAPthresh) 
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
  SHAP <- sapply(Pro_Plot$uniprotswissprot, function(x) {SHAP_PlotF[which(rownames(SHAP_PlotF)==x), "CombinedShap"]})
  Pro_Plot_F <- Pro_Plot %>% dplyr::select(hgnc_symbol) %>% mutate("SHAP" = SHAP) %>% arrange(desc(SHAP))
  
  Full_SHAP <- getBM(
    attributes = c("uniprotswissprot", "hgnc_symbol"),
    filters = "uniprotswissprot",
    values = Full_SHAP_Ori,
    mart = ensembl
  )
  SHAP_All <- sapply(Full_SHAP$uniprotswissprot, function(x) {Full_SHAP_F[which(rownames(Full_SHAP_F)==x), "SHAP"]})
  Full_SHAP_F_Plot <- cbind(Full_SHAP$hgnc_symbol, SHAP_All) %>% as.data.frame()
  
}else{
  Pro_Plot_F <- cbind(rownames(SHAP_PlotF), SHAP_PlotF$CombinedShap)
  Full_SHAP_F_Plot <- cbind(rownames(NewSHAP),NewSHAP) %>% as.data.frame()
}

###  Set the arbitrary colnames for the convenience of processing data.
colnames(Pro_Plot_F) <- colnames(Full_SHAP_F_Plot) <- c("proName", "SHAP")
rownames(Pro_Plot_F) <- Pro_Plot_F$proName

### Tackle with the same UniProtID corresponding to different Entrez Symbols
Full_SHAP_F_Plot %<>% group_by(proName) %>% summarise(SHAP = mean(SHAP, na.rm = TRUE), .groups = "drop")
rownames(Full_SHAP_F_Plot) <- Full_SHAP_F_Plot$proName

###############
# OUTPUT FILES
###############
### Output network plots and hub protein tables
pdf(paste0(outPath, titleMessage, "_Network.pdf"))
Hub_Proteins_STRING_List <- map2Srting(Pro_Plot_F, Full_SHAP_F_Plot, score_thresholdHere, combined_score_thresholdHere, titleMessage)
Hub_Proteins_STRING <- Hub_Proteins_STRING_List[[1]] %>% arrange(Closeness)
Hub_Proteins_STRING_WithExtention <- Hub_Proteins_STRING_List[[2]] %>% arrange(Closeness)

write.table(Hub_Proteins_STRING, file = paste0(outPath, titleMessage,"_Hub_Proteins_STRING.csv"))
write.table(Hub_Proteins_STRING_WithExtention, file = paste0(outPath, titleMessage, "_Hub_Proteins_STRING_WithExtention.csv"))
dev.off()

# png('PPI_M.png', res = 400)
# Hub_Proteins_GENIE <- regulatoryNet_WF(ProImportanceF,proDFhere,KnodeHere,nTreeHere,RFedge_threshold_here,titleMessage)
# Hub_Proteins_GENIE %<>% arrange(desc(Degree))
# write.table(Hub_Proteins_GENIE, file = "Hub_Proteins_GENIE.csv")
# dev.off()

# use the long flag
# Rscript Module_ProteinNetwork_WDL.R -s 400 -c 500 -R 0.04 -K 5 -n 1 -E proDFhere.csv > /Users/ydeng/Documents/IMCMCode/output/ProteinNetwork_WDL_log.csv 2>&1

# Rscript Module_ProteinNetwork_WDL.R -s 400 -c 500 -R 0.04 -K 5 -n 1 -E proDFhere.csv > /Users/ydeng/Documents/IMCMCode/output/ProteinNetwork_WDL_log.csv 2>&1
######################################################
######################################################
# For Yuhan to write in the paper only
pdf(paste0(outPath,"YH_paper_only.pdf"))
### The following is for Yuhan's 4 interested proteins
interestPro <- c("HPX", "APOH", "PLG", "GLRX")
ensembl <- useMart("ensembl", dataset = "hsapiens_gene_ensembl")
OverLap <- getBM(
  attributes = c("uniprotswissprot", "hgnc_symbol"),
  filters = "hgnc_symbol",
  values = interestPro,
  mart = ensembl
) %>% filter(uniprotswissprot != "")
rankInOurCombinedSHAP <- sapply(OverLap$uniprotswissprot, function(x) {which(rownames(NewSHAP)==x)})
rankInLightgbm <- sapply(OverLap$uniprotswissprot, function(x) {which(rownames(ProImportance[[1]])==x)})
rankInRF <- sapply(OverLap$uniprotswissprot, function(x) {which(rownames(ProImportance[[2]])==x)})
whereInterestPro <- cbind(OverLap, rankInLightgbm, rankInRF, rankInOurCombinedSHAP)
whereInterestPro

library(ggridges)
library(reshape2)  # For reshaping data
# Convert data to long format for ggridges
NewSHAP_long <- melt(NewSHAP, measure.vars = c("lightgbm", "random_forest", "CombinedShap"))

# Ridge plot
ggplot(NewSHAP_long, aes(x = value, y = variable, fill = variable)) +
  geom_density_ridges(alpha = 0.7, scale = 1.2) +
  labs(title = "Ridge Plot of SHAP Distributions", x = "SHAP Value", y = "Model") +
  theme_minimal() +
  scale_fill_manual(values = c("red", "blue", "green"))  # Custom colors


plot(NewSHAP$lightgbm,NewSHAP$random_forest, xlab="Lightgbm", ylab="random_forest", main="Comparisons of SHAP for all proteins")

dev.off()


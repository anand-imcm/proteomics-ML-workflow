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
library(clusterProfiler)
library(enrichplot)
library(org.Hs.eg.db)
library(gridExtra)
library(grid)

# Define the list of options
option_list <- list(
  make_option(c("-s", "--score_thresholdHere"), type = "numeric", default = 400, 
              help = "Score threshold for STRING database.", metavar = "SCORE"),
  make_option(c("-c", "--combined_score_thresholdHere"), type = "numeric", default = 500, 
              help = "Combined score threshold to select the nodes to be plotted.", metavar = "SCORE"),
  make_option(c("-R", "--RFedge_threshold_here"), type = "numeric", default = 0.04, 
              help = "RF edge threshold to be plotted.", metavar = "THRESHOLD"),
  make_option(c("-K", "--KnodeHere"), type = "integer", default = 10, 
              help = "Number of random nodes selected per iteration for RF.", metavar = "NODES"),
  make_option(c("-n", "--nTreeHere"), type = "integer", default = 10, 
              help = "Number of trees built per iteration for RF.", metavar = "TREE"),
  make_option(c("-t", "--titleMessage"), type = "character", default = "", 
              help = "Input title message for the output plots.", metavar = "MESSAGE"),
  make_option(c("-p","--proteinImportanceFile"), type = "character", default = "SVM_shap_values.csv", 
              help = "Protein importance input file name.", metavar = "IMPORTANCE"),
  make_option(c("-x","--proteinExpFile"), type = "character", default = "label_diagnosis_HCRBD2.csv", 
              help = "Protein expression input file name.", metavar = "EXPRESSION")
)

# Create an OptionParser object
parser <- OptionParser(option_list = option_list)

# Parse the command-line arguments
args <- parse_args(parser)

# Assign parsed values to variables
score_thresholdHere <- args$score_thresholdHere
combined_score_thresholdHere <- args$combined_score_thresholdHere
RFedge_threshold_here <- args$RFedge_threshold_here
KnodeHere <- args$KnodeHere
nTreeHere <- args$nTreeHere
titleMessage <- args$titleMessage
proteinImportanceFile <- args$proteinImportanceFile
proteinExpFile <- args$proteinExpFile

#--------------------------------
#--------------------------------
# FUNCTIONS TO BE CALLED BY MAIN
#--------------------------------
#--------------------------------
### Function to generate hub protein table based on centrality score
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

### Make network plot by mapping important proteins to STRING database
map2Srting_WF <- function(ProImportanceF,score_thresholdHere,combined_score_thresholdHere,titleMessage){
  
  string_db <- STRINGdb$new(version="12", species=9606, score_threshold=score_thresholdHere, network_type="full", input_directory="", protocol="http")
  
  proName <- rownames(ProImportanceF)
  
  ### initialize STRING database
  mapAll <- string_db$get_aliases()
  
  ### map STRINGID
  proNameF <- data.frame("prot"=proName)
  stDB_mapped <- string_db$map(proNameF,"prot",removeUnmappedRows = TRUE)
  
  ### retrieve interactions (avoid duplicated edges)
  interaction_network <- string_db$get_interactions(stDB_mapped$STRING_id)
  edges <- interaction_network[, c("from", "to", "combined_score")] %>% dplyr::filter(combined_score>combined_score_thresholdHere) %>% dplyr::distinct(from, to)
  
  ### construct igraph object
  g <- graph_from_data_frame(edges, directed = FALSE) 
  
  ### retrieve node name as protein symbol
  nodeName <- V(g)$name
  nodeName2 <-  mapAll$alias[unlist(sapply(nodeName,function(x){which(mapAll$STRING_id==x)[1]}))] ### because among the alias the first one corresponds to the protein symbol
  
  ### set node color
  nodeValue <- sapply(rownames(ProImportanceF),function(x){ifelse(x %in% nodeName2,ProImportanceF[x,]*100,NA)})
  
  nodeValueNorm <- sapply(nodeValue,function(x){
    if(!is.na(x)){
      x2 <- as.numeric(x)
      (x2 - min(nodeValue,na.rm=TRUE)) / (max(nodeValue,na.rm=TRUE) - min(nodeValue,na.rm=TRUE))
    }else{"grey"}
  })
  
  myPalette <- colorRampPalette(c("darkgreen","lightgreen","white","pink","darkred"))
  nodeColor <- nodeValueNorm
  nodeColor[which(nodeColor !="grey")] <- myPalette(1000)[as.numeric(cut(as.numeric(nodeColor[which(nodeColor !="grey")]), breaks=1000))]
  
  ### make the plot
  plot(g, vertex.label=nodeName2, vertex.size = 3, vertex.label.cex=0.5, vertex.color = nodeColor[nodeName2],edge.color="grey", vertex.label.color = "firebrick",
       vertex.label.family = "sans", vertex.label.dist=1, edge.arrow.size = 0.1, edge.arrow.color = "cyan",
       cex.main=0.06, main = paste0("Protein-Protein Interaction Network",titleMessage,"\nBased on STRING database"), rescale = TRUE)
  
  image.plot(legend.only=TRUE, zlim=range(as.numeric(nodeValue[which(!is.na(nodeValue))])/100), col=myPalette(1000),
             horizontal = TRUE,legend.width=0.3, legend.shrink=0.3, legend.mar=3, 
             legend.args=list(text="Feature Importance Score", side=1, font=2, line=1, cex=0.8),
             axis.args = list(cex.axis = 0.8,mgp = c(3, 0.3, 0)))
  
  ### Generate table of centrality score to display hub proteins
  HubProteins <- getHubProTable(edges)
  Hub_Proteins_STRING <- merge(HubProteins, stDB_mapped, by.x = "Protein", by.y = "STRING_id")
  Hub_Proteins_STRING %<>% mutate("Protein" = prot) %<>% dplyr::select(Protein,Betweenness,Closeness,Degree)
  
  return(Hub_Proteins_STRING)
}

### Make network plot by applying GENEI3 Random Forrest Algorithm bsaed on protein expression data
regulatoryNet_WF <- function(ProImportanceF,proDFhere,KnodeHere,nTreeHere,RFedge_threshold_here,titleMessage){
  ### expression profile of master regulators applied to GENIE3
  proName <- rownames(ProImportanceF)
  masterRexp = proDFhere[,intersect(proName,colnames(proDFhere))]
  
  ### construct the tree
  weightMat <- GENIE3(t(masterRexp), treeMethod="ET", K=KnodeHere, nTrees=nTreeHere)
  netRaw <- getLinkList(weightMat,threshold=RFedge_threshold_here)
  
  ### retrieve regulatory network nodes and edge information, plot network
  adjM <- as_adjacency_matrix(graph_from_edgelist(as.matrix(netRaw[,1:2]), directed=TRUE))
  gra <- graph_from_adjacency_matrix(adjM, mode = "directed", weighted = TRUE,diag = FALSE, add.colnames = NULL, add.rownames = NA)
  vertex.display <- V(gra)$name
  
  ### set node color
  nodeValue <- ProImportanceF[vertex.display,]
  names(nodeValue) <- vertex.display
  
  nodeColor <- nodeValueNorm <- sapply(nodeValue,function(x){
    (x - min(nodeValue)) / (max(nodeValue) - min(nodeValue))
  })
  
  myPalette <- colorRampPalette(c("darkgreen","lightgreen","white","pink","darkred"))
  nodeColor <- myPalette(1000)[as.numeric(cut(as.numeric(nodeColor), breaks=1000))]
  names(nodeColor) <- names(nodeValue)
  
  plot(gra, main=paste0("Protein Regulatory Network\n",titleMessage), cex.main=0.8,
       vertex.label=vertex.display,vertex.size=3, 
       vertex.color=nodeColor[vertex.display],vertex.label.font=0.5, vertex.label.cex=.5, vertex.label.color="darkgreen",
       edge.arrow.size=0.2, arrow.width=1, vertex.label.dist=1.5,rescale = TRUE)
  
  image.plot(legend.only=TRUE, zlim=range(nodeValue), col=myPalette(1000),
             horizontal = TRUE,legend.width=0.3, legend.shrink=0.3, legend.mar=3, 
             legend.args=list(text="Feature Importance Score", side=1, font=2, line=1, cex=0.5),
             axis.args = list(cex.axis = 0.5,mgp = c(3, 0.3, 0)))
  
  ### Generate table of centrality score to display hub proteins
  Hub_Proteins <- getHubProTable(netRaw)
  
  return(Hub_Proteins)
}

#--------
#--------
# MAIN
#--------
#--------
################
# READ IN DATA
################
ProImportanceF <- read.csv(proteinImportanceFile,row.names=1)
ProImportanceF %<>% filter(Mean.SHAP.Value!=0) %<>% arrange(desc(Mean.SHAP.Value))
impotantPro <- rownames(ProImportanceF)
SHAPvalue <- ProImportanceF$Mean.SHAP.Value
names(SHAPvalue) <- impotantPro
proDFhere <- read.table(proteinExpFile,sep=",",header=TRUE, row.names = 2)[,-1]
proUniverse <- colnames(proDFhere)

###############
# OUTPUT FILES
###############
### Output network plots and hub protein tables
png('PPI_Net.png',res = 400)
Hub_Proteins_STRING <- map2Srting_WF(ProImportanceF,score_thresholdHere,combined_score_thresholdHere,titleMessage)
Hub_Proteins_STRING %<>% arrange(Closeness)
write.table(Hub_Proteins_STRING, file = "Hub_Proteins_STRING.csv")
dev.off()

png('Regulatory_Net.png', res = 400)
Hub_Proteins_GENIE <- regulatoryNet_WF(ProImportanceF,proDFhere,KnodeHere,nTreeHere,RFedge_threshold_here,titleMessage)
Hub_Proteins_GENIE %<>% arrange(desc(Degree))
write.table(Hub_Proteins_GENIE, file = "Hub_Proteins_GENIE.csv")
dev.off()

### Pathway Enrichment Test
gmtList <- list.files(pattern = "gmt")

for(pathwayFile in gmtList){
  
  dataBase <- clusterProfiler::read.gmt(paste0(inputPath,pathwayFile))
  
  enrichPath <- clusterProfiler::enricher(gene = impotantPro, universe = proUniverse, pvalueCutoff = 1, qvalueCutoff = 0.05, pAdjustMethod = "BH",
                                          TERM2GENE=dataBase)
  topPath <- enrichPath@result %>% filter(p.adjust < 0.05)
  geneInPath <- unique(intersect(impotantPro,unique(dataBase[which(dataBase$term %in% rownames(topPath)),"gene"])))
  
  if(nrow(topPath) >1 ){
    
    ### Barchart showing significant enriched pathways
    p1 <- barplot(enrichPath, showCategory=5) + 
      theme(axis.text.y = element_text(size = 6, face = "bold"),
            axis.text.x = element_text(size = 5),
            axis.title.x = element_text(size =6, face = "bold"),
            legend.title = element_text(size = 6, face = "bold"),
            legend.text = element_text(size = 6))
    
    ### Pathway crosstalk
    p2 <- enrichplot::cnetplot(
      enrichPath,
      showCategory = min(3, nrow(topPath)), 
      # Parameters for color: foldChange (SHAPvalue) and edge coloring enabled
      color.params = list(foldChange = SHAPvalue, edge = FALSE), 
      cex.params = list(gene_label = 0.3, category_label = 0.4), 
      node_label = "all") + 
      scale_color_gradient2(name = "SHAP value", low = 'darkgreen', mid = "white" ,high = 'firebrick', midpoint = median(SHAPvalue[geneInPath])) +
      theme(
        legend.title = element_text(size = 5, face="bold"), 
        legend.text = element_text(size = 5)
      ) + guides(size = "none") 
    
    png(paste0(pathwayFile,".png"), res = 400)
    grid.arrange(p1, p2, heights=c(0.2, 0.8), nrow=2)
    dev.off()
    
  }else{print(paste0("No significant enriched pathway based on ", pathwayFile, " ."))}
}

# Rscript Module_ProteinNetwork_WDL.R -s 400 -c 500 -R 0.04 -K 5 -n 1 -E proDFhere.csv > /Users/ydeng/Documents/IMCMCode/output/ProteinNetwork_WDL_log.csv 2>&1



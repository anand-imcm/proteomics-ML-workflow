#!/bin/bash
# I did not use different output file names for different cases for the sake of succinct of output from main workflow
# Need to run the Rcript seperately not concurrently to replicate the results in the manuscipt
Rscript Module_ProteinNetwork_WDL.R \
    --inputPath "/home/rstudio/YD/ML_workflow/input/" \
    --outPath "/home/rstudio/YD/ML_workflow/output/" \
    --score_thresholdHere 400 \
    --combined_score_thresholdHere 800 \
    --SHAPthresh 100 \
    --patternChosen "Case1_lightgbm" \
    --converProId TRUE \
    --proteinExpFile "/home/rstudio/YD/ML_workflow/input/expression/Case1.csv" \
    --CorMethod "spearman" \
    --CorThreshold 0.8 \
    > "/home/rstudio/YD/ML_workflow/output/Network.log" 2>&1 


Rscript Module_ProteinNetwork_WDL.R \
    --inputPath "/home/rstudio/YD/ML_workflow/input/" \
    --outPath "/home/rstudio/YD/ML_workflow/output/" \
    --score_thresholdHere 400 \
    --combined_score_thresholdHere 800 \
    --SHAPthresh 100 \
    --patternChosen "Case1_random_forest" \
    --converProId TRUE \
    --proteinExpFile "/home/rstudio/YD/ML_workflow/input/expression/Case1.csv" \
    --CorMethod "spearman" \
    --CorThreshold 0.8 \
    >> "/home/rstudio/YD/ML_workflow/output/Network.log" 2>&1 


Rscript Module_ProteinNetwork_WDL.R \
    --inputPath "/home/rstudio/YD/ML_workflow/input/" \
    --outPath "/home/rstudio/YD/ML_workflow/output/" \
    --score_thresholdHere 400 \
    --combined_score_thresholdHere 800 \
    --SHAPthresh 100 \
    --patternChosen "Case1" \
    --converProId TRUE \
    --proteinExpFile "/home/rstudio/YD/ML_workflow/input/expression/Case1.csv" \
    --CorMethod "spearman" \
    --CorThreshold 0.8 \
    >> "/home/rstudio/YD/ML_workflow/output/Network.log" 2>&1 

Rscript Module_ProteinNetwork_WDL.R \
    --inputPath "/home/rstudio/YD/ML_workflow/input/" \
    --outPath "/home/rstudio/YD/ML_workflow/output/" \
    --score_thresholdHere 400 \
    --combined_score_thresholdHere 800 \
    --SHAPthresh 100 \
    --patternChosen "Case2_svm" \
    --converProId TRUE \
    --proteinExpFile "/home/rstudio/YD/ML_workflow/input/expression/Case2.csv" \
    --CorMethod "spearman" \
    --CorThreshold 0.8 \
    >> "/home/rstudio/YD/ML_workflow/output/Network.log" 2>&1 

Rscript Module_ProteinNetwork_WDL.R \
    --inputPath "/home/rstudio/YD/ML_workflow/input/" \
    --outPath "/home/rstudio/YD/ML_workflow/output/" \
    --score_thresholdHere 400 \
    --combined_score_thresholdHere 800 \
    --SHAPthresh 100 \
    --patternChosen "Case2_neural_network" \
    --converProId TRUE \
    --proteinExpFile "/home/rstudio/YD/ML_workflow/input/expression/Case2.csv" \
    --CorMethod "spearman" \
    --CorThreshold 0.8 \
    >> "/home/rstudio/YD/ML_workflow/output/Network.log" 2>&1 

Rscript Module_ProteinNetwork_WDL.R \
    --inputPath "/home/rstudio/YD/ML_workflow/input/" \
    --outPath "/home/rstudio/YD/ML_workflow/output/" \
    --score_thresholdHere 400 \
    --combined_score_thresholdHere 800 \
    --SHAPthresh 100 \
    --patternChosen "Case2" \
    --converProId TRUE \
    --proteinExpFile "/home/rstudio/YD/ML_workflow/input/expression/Case2.csv" \
    --CorMethod "spearman" \
    --CorThreshold 0.8 \
    >> "/home/rstudio/YD/ML_workflow/output/Network.log" 2>&1      
#!/bin/bash

Rscript Module_ProteinNetwork_WDL.R \
    --inputPath "/home/rstudio/YD/ML_workflow/input/" \
    --outPath "/home/rstudio/YD/ML_workflow/output/" \
    --score_thresholdHere 400 \
    --combined_score_thresholdHere 800 \
    --SHAPthresh 100 \
    --patternChosen "Case1_lightgbm" \
    --converProId TRUE \
    > "/home/rstudio/YD/ML_workflow/output/Network.log" 2>&1 &


Rscript Module_ProteinNetwork_WDL.R \
    --inputPath "/home/rstudio/YD/ML_workflow/input/" \
    --outPath "/home/rstudio/YD/ML_workflow/output/" \
    --score_thresholdHere 400 \
    --combined_score_thresholdHere 800 \
    --SHAPthresh 100 \
    --patternChosen "Case1_random_forest" \
    --converProId TRUE \
    >> "/home/rstudio/YD/ML_workflow/output/Network.log" 2>&1 &


Rscript Module_ProteinNetwork_WDL.R \
    --inputPath "/home/rstudio/YD/ML_workflow/input/" \
    --outPath "/home/rstudio/YD/ML_workflow/output/" \
    --score_thresholdHere 400 \
    --combined_score_thresholdHere 800 \
    --SHAPthresh 100 \
    --patternChosen "Case1" \
    --converProId TRUE \
    >> "/home/rstudio/YD/ML_workflow/output/Network.log" 2>&1 &

Rscript Module_ProteinNetwork_WDL.R \
    --inputPath "/home/rstudio/YD/ML_workflow/input/" \
    --outPath "/home/rstudio/YD/ML_workflow/output/" \
    --score_thresholdHere 400 \
    --combined_score_thresholdHere 800 \
    --SHAPthresh 100 \
    --patternChosen "Case2_svm" \
    --converProId TRUE \
    >> "/home/rstudio/YD/ML_workflow/output/Network.log" 2>&1 &

Rscript Module_ProteinNetwork_WDL.R \
    --inputPath "/home/rstudio/YD/ML_workflow/input/" \
    --outPath "/home/rstudio/YD/ML_workflow/output/" \
    --score_thresholdHere 400 \
    --combined_score_thresholdHere 800 \
    --SHAPthresh 100 \
    --patternChosen "Case2_neural_network" \
    --converProId TRUE \
    >> "/home/rstudio/YD/ML_workflow/output/Network.log" 2>&1 &

Rscript Module_ProteinNetwork_WDL.R \
    --inputPath "/home/rstudio/YD/ML_workflow/input/" \
    --outPath "/home/rstudio/YD/ML_workflow/output/" \
    --score_thresholdHere 400 \
    --combined_score_thresholdHere 800 \
    --SHAPthresh 100 \
    --patternChosen "Case2" \
    --converProId TRUE \
    >> "/home/rstudio/YD/ML_workflow/output/Network.log" 2>&1      
    

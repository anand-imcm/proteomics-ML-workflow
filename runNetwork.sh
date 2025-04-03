#!/bin/bash
### Script used to generate plots in the manuscript based on the two cases
Rscript Module_ProteinNetwork_WDL.R \
    --score_thresholdHere 400 \
    --combined_score_thresholdHere 800 \
    --SHAPthresh 100 \
    --patternChosen "shap_values.csv" \
    --converProId TRUE \
    --proteinExpFile "Case1.csv" \
    --CorMethod "spearman" \
    --CorThreshold 0.8 \
    > "Network.log" 2>&1 

# Rscript Module_ProteinNetwork_WDL.R \
#     --score_thresholdHere 400 \
#     --combined_score_thresholdHere 800 \
#     --SHAPthresh 50 \
#     --patternChosen "shap_values.csv" \
#     --converProId TRUE \
#     --proteinExpFile "Case2.csv" \
#     --CorMethod "spearman" \
#     --CorThreshold 0.8 \
#     >> "Network.log" 2>&1
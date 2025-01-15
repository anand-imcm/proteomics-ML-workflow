#!/usr/bin/env python3
# regression.py
# This script allows parallel execution of three separate scripts:
#   1) Step2_Regression_workflow.py (supports feature selection)
#   2) Step2_VAE_MLP_reg.py (no feature selection parameter)
#   3) Step2_MLP_in_VAE_reg.py (no feature selection parameter)
#
# Usage example:
#   python regression.py -i input.csv -p result_prefix -m NN_reg RF_reg VAE_reg -f pca
#
# Model name mapping for the -m parameter:
#   NN_reg       -> Neural_Network_reg
#   RF_reg       -> Random_Forest_reg
#   SVM_reg      -> SVM_reg
#   XGB_reg      -> XGBoost_reg
#   PLS_reg      -> PLS_reg
#   KNN_reg      -> KNN_reg
#   LightGBM_reg -> LightGBM_reg
#   VAE_reg      -> VAE_MLP_reg (Step2_VAE_MLP_reg.py)
#   MLPVAE_reg   -> MLP_in_VAE_reg (Step2_MLP_in_VAE_reg.py)

import argparse
import subprocess
import sys
from joblib import Parallel, delayed

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run multiple regression models (including VAE-based) in parallel."
    )

    parser.add_argument(
        "-i", "--csv", 
        type=str, 
        required=True, 
        help="Input CSV file with at least columns: SampleID, Label, and features."
    )
    parser.add_argument(
        "-p", "--prefix", 
        type=str, 
        required=True, 
        help="Output prefix for results."
    )
    parser.add_argument(
        "-m", "--models", 
        type=str, 
        nargs="+", 
        required=True,
        help=(
            "List of model shortcuts to run in parallel. Possible values: "
            "NN_reg, RF_reg, SVM_reg, XGB_reg, PLS_reg, KNN_reg, LightGBM_reg, "
            "VAE_reg, MLPVAE_reg"
        )
    )
    parser.add_argument(
        "-f", "--feature_selection", 
        type=str, 
        default="none",
        choices=["none", "elasticnet", "pca", "kpca", "umap", "pls", "tsne"],
        help="Feature selection method to use (ignored by VAE_reg and MLPVAE_reg)."
    )
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Map simplified model names to their actual names in Step2_Regression_workflow.py
    # (for the models that support feature selection)
    workflow_model_map = {
        "NN_reg": "Neural_Network_reg",
        "RF_reg": "Random_Forest_reg",
        "SVM_reg": "SVM_reg",
        "XGB_reg": "XGBoost_reg",
        "PLS_reg": "PLS_reg",
        "KNN_reg": "KNN_reg",
        "LightGBM_reg": "LightGBM_reg",
    }

    # Models handled by Step2_VAE_MLP_reg.py
    # (do not support the -f feature_selection parameter)
    vae_mlp_model_map = {
        "VAE_reg": "VAE_MLP_reg",
    }

    # Models handled by Step2_MLP_in_VAE_reg.py
    # (do not support the -f feature_selection parameter)
    mlp_in_vae_model_map = {
        "MLPVAE_reg": "MLP_in_VAE_reg",
    }

    # Validate requested models
    valid_models = set(list(workflow_model_map.keys()) +
                       list(vae_mlp_model_map.keys()) +
                       list(mlp_in_vae_model_map.keys()))

    # Check for invalid models
    for m in args.models:
        if m not in valid_models:
            print(f"Error: Invalid model '{m}'.")
            sys.exit(1)

    # Prepare parallel jobs
    jobs = []
    for m in args.models:
        if m in workflow_model_map:
            # Step2_Regression_workflow.py call
            cmd = [
                "python", "Step2_Regression_workflow.py",
                "-i", args.csv,
                "-p", args.prefix,
                "-m", workflow_model_map[m],
                "-f", args.feature_selection
            ]
            job_str = " ".join(cmd)
            jobs.append(job_str)
        elif m in vae_mlp_model_map:
            # Step2_VAE_MLP_reg.py call
            cmd = [
                "python", "Step2_VAE_MLP_reg.py",
                "-i", args.csv,
                "-p", args.prefix
            ]
            job_str = " ".join(cmd)
            jobs.append(job_str)
        elif m in mlp_in_vae_model_map:
            # Step2_MLP_in_VAE_reg.py call
            cmd = [
                "python", "Step2_MLP_in_VAE_reg.py",
                "-i", args.csv,
                "-p", args.prefix
            ]
            job_str = " ".join(cmd)
            jobs.append(job_str)

    # Run in parallel
    if not jobs:
        print("No valid jobs to run.")
        sys.exit(0)

    print("Running jobs in parallel:")
    for j in jobs:
        print(f"  {j}")

    # Use joblib's Parallel to run them
    # Each job is run via subprocess.run with shell=True
    Parallel(n_jobs=-1)(
        delayed(subprocess.run)(job, shell=True, check=True) for job in jobs
    )

    print("All jobs finished successfully.")

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
# regression.py
# This script allows parallel execution of three separate scripts:
#   1) Step2_Regression_workflow.py (supports multiple models with feature selection)
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
import os
from PIL import Image
import shlex

script_dir = os.path.dirname(os.path.abspath(__file__))

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
            "NN_reg, RF_reg, SVM_reg, XGB_reg, PLS_reg, KNN_reg, LGBM_reg, "
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
    workflow_model_map = {
        "NN_reg": "Neural_Network_reg",
        "RF_reg": "Random_Forest_reg",
        "SVM_reg": "SVM_reg",
        "XGB_reg": "XGBoost_reg",
        "PLS_reg": "PLS_reg",
        "KNN_reg": "KNN_reg",
        "LGBM_reg": "LightGBM_reg",
    }

    # Models handled by Step2_VAE_MLP_reg.py
    vae_mlp_model_map = {
        "VAE_reg": "VAE_MLP_reg",
    }

    # Models handled by Step2_MLP_in_VAE_reg.py
    mlp_in_vae_model_map = {
        "MLPVAE_reg": "MLP_in_VAE_reg",
    }

    # Validate requested models (case-insensitive normalization)
    valid_models = set(list(workflow_model_map.keys()) +
                       list(vae_mlp_model_map.keys()) +
                       list(mlp_in_vae_model_map.keys()))

    user_choices = args.models[:]
    for i, m in enumerate(user_choices):
        for choice in valid_models:
            if m.casefold() == choice.casefold():
                user_choices[i] = choice
                break

    # Separate models based on the script they should be handled by
    workflow_models = [workflow_model_map[m] for m in user_choices if m in workflow_model_map]
    vae_mlp_models = [vae_mlp_model_map[m] for m in user_choices if m in vae_mlp_model_map]
    mlp_in_vae_models = [mlp_in_vae_model_map[m] for m in user_choices if m in mlp_in_vae_model_map]

    # Prepare parallel jobs as argument lists (not strings)
    jobs = []

    if workflow_models:
        regression_workflow = os.path.join(script_dir, "Step2_Regression_workflow.py")
        cmd = ["python", regression_workflow, "-i", args.csv, "-p", args.prefix, "-m"] + workflow_models
        if args.feature_selection != "none":
            cmd += ["-f", args.feature_selection]
        jobs.append(cmd)

    for _ in vae_mlp_models:
        vae_mlp_reg = os.path.join(script_dir, "Step2_VAE_MLP_reg.py")
        cmd = ["python", vae_mlp_reg, "-i", args.csv, "-p", args.prefix]
        jobs.append(cmd)

    for _ in mlp_in_vae_models:
        mlp_in_vae_reg = os.path.join(script_dir, "Step2_MLP_in_VAE_reg.py")
        cmd = ["python", mlp_in_vae_reg, "-i", args.csv, "-p", args.prefix]
        jobs.append(cmd)

    if not jobs:
        print("No valid jobs to run.")
        sys.exit(0)

    print("Running jobs in parallel:")
    for j in jobs:
        # Pretty-print with proper quoting for paths containing spaces
        print(f"  {shlex.join(j)}")

    try:
        # Pass argument lists; do not use shell=True
        Parallel(n_jobs=-1)(
            delayed(subprocess.run)(job, check=True) for job in jobs
        )
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running jobs: {e}")
        sys.exit(1)

    print("All jobs finished successfully.")

    # Attempt to combine _models_summary_bar_chart.png, _vaemlp_reg_average_metrics.png and _vae_reg_average_metrics.png into one image
    try:
        image_names = [
            f"{args.prefix}_models_summary_bar_chart.png",
            f"{args.prefix}_vaemlp_reg_average_metrics.png",
            f"{args.prefix}_vae_reg_average_metrics.png",
        ]

        missing_images = [img for img in image_names if not os.path.exists(img)]
        if missing_images:
            print(
                "Warning: The following images were not found and will not be included in the combined summary: "
                + ", ".join(missing_images)
            )

        existing_images = [img for img in image_names if os.path.exists(img)]
        if not existing_images:
            print("No images available to combine.")
            sys.exit(0)

        images = [Image.open(img) for img in existing_images]

        widths, heights = zip(*(img.size for img in images))
        total_width = sum(widths)
        max_height = max(heights)

        combined_img = Image.new("RGB", (total_width, max_height), color=(255, 255, 255))
        x_offset = 0
        for im in images:
            combined_img.paste(im, (x_offset, 0))
            x_offset += im.size[0]

        combined_output = f"{args.prefix}_combined_summary.png"
        combined_img.save(combined_output)
        print(f"Combined summary image saved as {combined_output}")

    except Exception as e:
        print(f"Error combining images: {e}")

if __name__ == "__main__":
    main()

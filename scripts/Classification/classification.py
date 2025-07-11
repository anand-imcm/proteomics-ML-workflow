import argparse
import sys
from pathlib import Path
from joblib import Parallel, delayed
from Step2_RF import random_forest_nested_cv as random_forest
from Step2_KNN import knn_nested_cv as knn
from Step2_NN import neural_network_nested_cv as neural_network
from Step2_SVM import svm_nested_cv as svm
from Step2_XGBOOST import xgboost_nested_cv as xgboost
from Step2_PLSDA import plsda_nested_cv as plsda
from Step2_VAE_MLP import vae
from Step2_Light_GBM import lightgbm_nested_cv as lightgbm
from Step2_LR import logistic_regression_nested_cv as logistic_regression
from Step2_MLP_in_VAE import vae as mlpvae
from Step2_NB import gaussian_nb_nested_cv as gaussian_nb


def parse_arguments():
    parser = argparse.ArgumentParser(description='Script to run classifiers')
    parser.add_argument('-i', '--csv', type=str, required=True, help='Input file in CSV format')
    parser.add_argument('-m', '--model', type=str, nargs='+', default=['KNN', 'RF', 'NN', 'SVM', 'XGB', 'PLSDA', 'VAE', 'LGBM', 'LR', 'MLPVAE', 'GNB'],
                        choices=['KNN', 'RF', 'NN', 'SVM', 'XGB', 'PLSDA', 'VAE', 'LGBM', 'LR', 'MLPVAE', 'GNB'],
                        help='Name of the model(s), default is all models')
    parser.add_argument('-p', '--prefix', type=str, help='Output prefix')
    parser.add_argument('-f', '--feature_selection', type=str,
                        choices=['none', 'elasticnet', 'pca', 'kpca', 'umap', 'pls', 'tsne'],
                        help='Feature selection method (not applicable for VAE and MLPVAE)')
    return parser.parse_args()


def run_model(model, csv, prefix, feature_selection):
    if model == "RF":
        random_forest(csv, prefix, feature_selection)
    elif model == "KNN":
        knn(csv, prefix, feature_selection)
    elif model == "NN":
        neural_network(csv, prefix, feature_selection)
    elif model == "SVM":
        svm(csv, prefix, feature_selection)
    elif model == "XGB":
        xgboost(csv, prefix, feature_selection)
    elif model == "PLSDA":
        if feature_selection not in ['umap', 'elasticnet', None]:
            print(f"Skipping PLSDA because feature selection {feature_selection} is incompatible.")
            return
        plsda(csv, prefix, feature_selection)

    elif model == "VAE":
        # VAE does not take feature_selection
        vae(csv, prefix)
    elif model == "LGBM":
        lightgbm(csv, prefix, feature_selection)
    elif model == "LR":
        logistic_regression(csv, prefix, feature_selection)
    elif model == "MLPVAE":
        # MLPVAE does not take feature_selection
        mlpvae(csv, prefix)
    elif model == "GNB":
        gaussian_nb(csv, prefix, feature_selection)
    else:
        raise ValueError(f"Unsupported model: {model}")

    print(f"Finished {model}")


def main():
    args = parse_arguments()

    # Derive prefix from CSV filename if not provided
    prefix = Path(args.csv).stem
    if args.prefix:
        prefix = args.prefix

    if args.feature_selection.lower() == 'none':
        args.feature_selection = None

    # Validate feature selection method for PLSDA
    # Validate feature selection method for PLSDA
    if 'PLSDA' in args.model:
        if args.feature_selection not in ['umap', 'elasticnet', 'none', None]:
            print("Skipping PLSDA because the selected feature selection method is not compatible.")
            args.model.remove('PLSDA')


    # If feature selection is used, skip VAE and MLPVAE
    if args.feature_selection is not None:
        if "VAE" in args.model:
            print("Skipping VAE because feature selection is used.")
            args.model.remove("VAE")
        if "MLPVAE" in args.model:
            print("Skipping MLPVAE because feature selection is used.")
            args.model.remove("MLPVAE")

    # Run models in parallel
    Parallel(n_jobs=-1, backend='multiprocessing', verbose=100)(
        delayed(run_model)(model, args.csv, prefix, args.feature_selection) for model in args.model
    )


if __name__ == "__main__":
    main()

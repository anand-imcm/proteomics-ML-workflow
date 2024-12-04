import argparse
import sys
from pathlib import Path
import subprocess
sys.path.append((str(Path(__file__).resolve().parent)))
from joblib import Parallel, delayed
from Step2_RF import random_forest
from Step2_KNN import knn
from Step2_NN import neural_network
from Step2_SVM import svm
from Step2_XGBOOST import xgboost
from Step2_PLSDA import plsda
from Step2_VAE_MLP import vae
from Step2_Light_GBM import lightgbm
from Step2_LR import logistic_regression
from Step2_MLP_in_VAE import vae as mlpvae
from Step2_NB import gaussian_nb

def parse_arguments():
    parser = argparse.ArgumentParser(description='Script to run classifiers')
    parser.add_argument('-i','--csv',type=str, help='Input file in CSV format', required=True)
    parser.add_argument('-m', '--model', type=str, nargs='+', choices=['KNN', 'RF', 'NN', 'SVM', 'XGB', 'PLSDA', 'VAE', 'LGBM', 'LR', 'MLPVAE', 'NB'], help='Name of the model(s)', required=True)
    parser.add_argument('-p','--prefix',type=str, help='Output prefix')
    return parser.parse_args()

def run_model(model, csv, prefix):
    if model == "RF":
        random_forest(csv, prefix)
    elif model == "KNN":
        knn(csv, prefix)
    elif model == "NN":
        neural_network(csv, prefix)
    elif model == "SVM":
        svm(csv, prefix)
    elif model == "XGB":
        xgboost(csv, prefix)
    elif model == "PLSDA":
        plsda(csv, prefix)
    elif model == "VAE":
        vae(csv, prefix)
    elif model == "LGBM":
        lightgbm(csv, prefix)
    elif model == "LR":
        logistic_regression(csv, prefix)
    elif model == "MLPVAE":
        mlpvae(csv, prefix)
    elif model == "NB":
        gaussian_nb(csv, prefix)
    print(f"Finished {model}")

if __name__ == "__main__":
    args = parse_arguments()
    prefix = Path(args.csv).stem
    if args.prefix:
        prefix = args.prefix
    results = Parallel(n_jobs=-1, backend='multiprocessing', verbose=100)(
        delayed(run_model)(model, args.csv, prefix) for model in args.model
    )

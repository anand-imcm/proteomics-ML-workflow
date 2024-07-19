import argparse
from pathlib import Path
from .Step2_RF import random_forest
from .Step2_KNN import knn

def parse_arguments():
    parser = argparse.ArgumentParser(description='Script to run classifiers')
    parser.add_argument('-i','--csv',type=str, help='Input file in CSV format', required=True)
    parser.add_argument('-m','--model',type=str, choices=['KNN','RF'], help='Name of the model', required=True)
    parser.add_argument('-p','--prefix',type=str, help='Output prefix')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    prefix = Path(args.csv).stem
    if args.prefix:
        prefix = args.prefix
    prefix = Path(args.csv).stem
    if args.model == "RF":
        random_forest(args.csv,prefix)
    elif args.model == "KNN":
        knn(args.csv, prefix)
    print (f"Finished {args.model}")

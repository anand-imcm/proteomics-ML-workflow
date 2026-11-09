import argparse
import sys
from pathlib import Path

from sklearn.utils.parallel import Parallel, delayed

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.publication_utils import (
    DEFAULT_RANDOM_STATE,
    configure_publication_plots,
    set_global_seed,
    task_run_context,
)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run BiomarkerML classification models.")
    parser.add_argument("--input", "-i", "--csv", dest="csv", required=True, help="Input CSV file")
    parser.add_argument(
        "-m", "--model", nargs="+",
        default=["KNN", "RF", "NN", "SVM", "XGB", "PLSDA", "VAE", "LGBM", "LR", "MLPVAE", "GNB"],
        choices=["KNN", "RF", "NN", "SVM", "XGB", "PLSDA", "VAE", "LGBM", "LR", "MLPVAE", "GNB"],
    )
    parser.add_argument("-p", "--prefix", default=None)
    parser.add_argument("-f", "--feature_selection", default="no", help="yes/no automatic protein feature selection")
    parser.add_argument("--outdir", default=".", help="Base output directory")
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE)
    return parser.parse_args()


def run_model(model, csv, prefix, feature_selection):
    from engines.classification import run_model as run_engine_model

    run_engine_model(model, csv, prefix, feature_selection)


def main():
    configure_publication_plots()
    args = parse_arguments()
    set_global_seed(args.seed)
    prefix = args.prefix or Path(args.csv).stem
    with task_run_context(args.outdir, "classification"):
        Parallel(n_jobs=args.jobs, verbose=50)(
            delayed(run_model)(model, args.csv, prefix, args.feature_selection) for model in args.model
        )


if __name__ == "__main__":
    main()

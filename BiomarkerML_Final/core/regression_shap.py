from __future__ import annotations

try:
    from .workflow_tasks import common_parser, run_regression_shap
except ImportError:
    from workflow_tasks import common_parser, run_regression_shap


def main() -> None:
    parser = common_parser("Run BiomarkerML regression SHAP task.")
    parser.add_argument("--models", nargs="*", default=None, choices=["NN_reg", "RF_reg", "SVM_reg", "XGB_reg", "PLS_reg", "KNN_reg", "LGBM_reg", "VAE_reg", "MLPVAE_reg"])
    parser.add_argument("--num_features", type=int, default=20)
    run_regression_shap(parser.parse_args())


if __name__ == "__main__":
    main()

from __future__ import annotations

try:
    from .runtime_warnings import run_with_runtime_output_filter
    from .workflow_tasks import common_parser, run_regression
except ImportError:
    from runtime_warnings import run_with_runtime_output_filter
    from workflow_tasks import common_parser, run_regression


def main() -> None:
    run_with_runtime_output_filter(_main)


def _main() -> None:
    parser = common_parser("Run BiomarkerML regression task.")
    parser.add_argument("--models", nargs="*", default=None, choices=["NN_reg", "RF_reg", "SVM_reg", "XGB_reg", "PLS_reg", "KNN_reg", "LGBM_reg", "VAE_reg", "MLPVAE_reg"])
    run_regression(parser.parse_args())


if __name__ == "__main__":
    main()

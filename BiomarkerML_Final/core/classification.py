from __future__ import annotations

try:
    from .runtime_warnings import run_with_runtime_output_filter
    from .workflow_tasks import common_parser, run_classification
except ImportError:
    from runtime_warnings import run_with_runtime_output_filter
    from workflow_tasks import common_parser, run_classification


def main() -> None:
    run_with_runtime_output_filter(_main)


def _main() -> None:
    parser = common_parser("Run BiomarkerML classification task.")
    parser.add_argument("--models", nargs="*", default=None, choices=["KNN", "RF", "NN", "SVM", "XGB", "PLSDA", "VAE", "LGBM", "LR", "MLPVAE", "GNB"])
    run_classification(parser.parse_args())


if __name__ == "__main__":
    main()

from __future__ import annotations

try:
    from .workflow_tasks import common_parser, run_classification_shap
except ImportError:
    from workflow_tasks import common_parser, run_classification_shap


def main() -> None:
    parser = common_parser("Run BiomarkerML classification SHAP task.")
    parser.add_argument("--models", nargs="*", default=None, choices=["KNN", "RF", "NN", "SVM", "XGB", "PLSDA", "VAE", "LGBM", "LR", "MLPVAE", "GNB"])
    parser.add_argument("--num_features", type=int, default=20)
    run_classification_shap(parser.parse_args())


if __name__ == "__main__":
    main()

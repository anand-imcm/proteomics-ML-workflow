from __future__ import annotations

try:
    from .runtime_warnings import run_with_runtime_output_filter
    from .workflow_tasks import common_parser, run_preprocess
except ImportError:
    from runtime_warnings import run_with_runtime_output_filter
    from workflow_tasks import common_parser, run_preprocess


def main() -> None:
    run_with_runtime_output_filter(_main)


def _main() -> None:
    parser = common_parser("Run BiomarkerML preprocessing and visualization.")
    parser.add_argument("--methods", nargs="*", default=None, choices=["pca", "kpca", "umap", "tsne", "pls", "elasticnet", "boruta"])
    parser.add_argument("--dimensions", type=int, default=3)
    run_preprocess(parser.parse_args())


if __name__ == "__main__":
    main()

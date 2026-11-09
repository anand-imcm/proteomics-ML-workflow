from __future__ import annotations

try:
    from .workflow_tasks import common_parser, run_biomarker_reliability
except ImportError:
    from workflow_tasks import common_parser, run_biomarker_reliability


def main() -> None:
    parser = common_parser("Run BiomarkerML biomarker reliability task.")
    parser.add_argument("--task", choices=["classification", "regression"], default="classification")
    parser.add_argument("--dataset", default=None, help="Original input CSV used for publication summary metadata.")
    run_biomarker_reliability(parser.parse_args())


if __name__ == "__main__":
    main()

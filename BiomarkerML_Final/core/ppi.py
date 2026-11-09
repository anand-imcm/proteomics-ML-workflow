from __future__ import annotations

try:
    from .workflow_tasks import common_parser, run_ppi
except ImportError:
    from workflow_tasks import common_parser, run_ppi


def main() -> None:
    parser = common_parser("Run BiomarkerML PPI task.")
    parser.add_argument("--protein_expression", default=None)
    parser.add_argument("--species", type=int, default=9606)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--top_n", type=int, default=100)
    parser.add_argument("--score_threshold", type=int, default=400)
    parser.add_argument("--combined_score_threshold", type=int, default=800)
    parser.add_argument("--expanded_limit", type=int, default=5)
    parser.add_argument("--correlation_method", default="spearman", choices=["spearman", "pearson", "kendall"])
    parser.add_argument("--correlation_threshold", type=float, default=0.8)
    run_ppi(parser.parse_args())


if __name__ == "__main__":
    main()

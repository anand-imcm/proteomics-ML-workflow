from __future__ import annotations

try:
    from .workflow_tasks import common_parser, run_pdf_report
except ImportError:
    from workflow_tasks import common_parser, run_pdf_report


def main() -> None:
    parser = common_parser("Build BiomarkerML PDF report.")
    run_pdf_report(parser.parse_args())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from core.runtime_warnings import run_with_runtime_output_filter

ROOT = Path(__file__).resolve().parent
CORE = ROOT / "core"
PAPER_TOOLS = ROOT / "paper_tools"

COMMANDS = {
    "run": CORE / "workflow.py",
    "preprocess": CORE / "preprocess.py",
    "classify": CORE / "classification.py",
    "regress": CORE / "regression.py",
    "shap-classification": CORE / "classification_shap.py",
    "shap-regression": CORE / "regression_shap.py",
    "reliability": CORE / "biomarker_reliability.py",
    "ppi": CORE / "ppi.py",
    "report": CORE / "report.py",
    "paper-benchmark": PAPER_TOOLS / "benchmark_nested_vs_standard" / "benchmark.py",
    "paper-figures": PAPER_TOOLS / "publication_figure_assembly.py",
}


def main() -> None:
    run_with_runtime_output_filter(_main)


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="BiomarkerML command registry.",
        epilog="Use 'run' for the canonical end-to-end workflow. Task commands accept --profile {safe,balanced,aggressive,expert}.",
    )
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to the selected command")
    ns = parser.parse_args()
    script = COMMANDS[ns.command]
    if not script.exists():
        raise SystemExit(f"Command target not found: {script}")
    forwarded = ns.args[1:] if ns.args[:1] == ["--"] else ns.args
    env = dict(os.environ)
    pythonpath = ":".join([str(ROOT), str(CORE)])
    env["PYTHONPATH"] = pythonpath + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    raise SystemExit(subprocess.call([sys.executable, str(script), *forwarded], env=env))


if __name__ == "__main__":
    main()

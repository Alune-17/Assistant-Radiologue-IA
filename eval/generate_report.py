from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from eval.reporting import generate_evaluation_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Génère un rapport Markdown à partir des fichiers CSV produits par eval/run_evaluation.py."
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "eval" / "outputs")
    parser.add_argument("--report-path", type=Path, default=ROOT / "eval" / "outputs" / "evaluation_report.md")
    args = parser.parse_args()

    report_path = generate_evaluation_report(args.out_dir, args.report_path)
    print(report_path)


if __name__ == "__main__":
    main()

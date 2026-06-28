from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from eval.case_review import build_case_review_template


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Génère le CSV de revue humaine pour les 20 à 30 cas commentés."
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "eval" / "outputs")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    path = build_case_review_template(args.out_dir, args.output, limit=args.limit)
    if path is None:
        raise SystemExit(
            "Aucune prédiction trouvée. Lancez d'abord : python eval/run_evaluation.py --mode toy"
        )
    print(path)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.inference import toy_predict, gemini_predict_baseline, gemini_predict_improved
from src.guardrails import apply_safety_guardrails, validate_prediction
from src.metrics import summarize_metrics
from src.database import insert_run, init_db

# Mapping mode → fonction d'inférence
INFERENCE_MAP = {
    "baseline": lambda p: toy_predict(p, mode="baseline"),
    "improved":  lambda p: toy_predict(p, mode="improved"),
    "gemini-baseline": gemini_predict_baseline,
    "gemini-improved": gemini_predict_improved,
}


def read_cases(path: Path) -> list[dict]:
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def run(mode: str, db_path: Path) -> tuple[list[dict], dict]:
    """Exécute l'évaluation complète sur les 30 cas synthétiques.

    Args:
        mode: Mode d'inférence ('baseline', 'improved', 'gemini-baseline', 'gemini-improved').
        db_path: Chemin vers la base SQLite pour les logs.

    Returns:
        Tuple (lignes de prédictions, dictionnaire de métriques).
    """
    cases = read_cases(ROOT / 'data' / 'synthetic_cases.csv')
    predict_fn = INFERENCE_MAP[mode]
    rows = []
    init_db(db_path)

    print(f"\n[>>] Evaluation mode '{mode}' sur {len(cases)} cas...", file=sys.stderr)
    for i, case in enumerate(cases, 1):
        image_path = ROOT / case['image_path']
        pred = apply_safety_guardrails(predict_fn(image_path))
        valid, errors = validate_prediction(pred)
        row = {
            'case_id': case['case_id'],
            'label': case['label'],
            'predicted_class': pred['predicted_class'],
            'confidence': pred['confidence'],
            'image_quality': pred.get('image_quality', ''),
            'json_valid': valid,
            'warning': pred.get('warning', ''),
            'latency_ms': pred.get('latency_ms', 0),
            'guardrail_errors': ';'.join(errors),
            'model_name': pred.get('model_name', mode),
            'prompt_version': pred.get('prompt_version', mode),
        }
        rows.append(row)
        insert_run(db_path, case['case_id'], str(image_path), pred)

        # Affichage de progression (stderr uniquement)
        match = "[OK]" if row['label'] == row['predicted_class'] else "[ERR]"
        print(f"  [{i:02d}/{len(cases)}] {match} {case['case_id']:30s} | label={row['label']:20s} | pred={row['predicted_class']:20s} | conf={row['confidence']:.2f} | {row['latency_ms']}ms", file=sys.stderr)

    metrics = summarize_metrics(rows)
    return rows, metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Évaluation du prototype radiologue virtuel sur les cas synthétiques."
    )
    parser.add_argument(
        '--mode',
        choices=['toy', 'baseline', 'improved', 'gemini-baseline', 'gemini-improved', 'all-gemini'],
        default='baseline',
        help=(
            "'baseline'/'improved' : toy deterministe. "
            "'gemini-baseline'/'gemini-improved' : vrais appels API Gemini. "
            "'all-gemini' : compare les deux modes Gemini. "
            "'toy' : alias pour baseline+improved (toy)."
        ),
    )
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'eval' / 'outputs')
    parser.add_argument('--db-path', type=Path, default=ROOT / 'medical_ai_evidence.sqlite')
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sélection des modes à exécuter
    if args.mode == 'toy':
        modes = ['baseline', 'improved']
    elif args.mode == 'all-gemini':
        modes = ['gemini-baseline', 'gemini-improved']
    else:
        modes = [args.mode]

    summary = []
    for mode in modes:
        rows, metrics = run(mode, args.db_path)
        write_csv(out_dir / f'{mode}_predictions.csv', rows)
        (out_dir / f'{mode}_metrics.json').write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8'
        )
        summary.append({'mode': mode, **metrics})
        print(f"\n[>>] Metriques {mode} :", file=sys.stderr)
        for k, v in metrics.items():
            print(f"   {k}: {v}", file=sys.stderr)

    write_csv(out_dir / 'before_after_summary.csv', summary)
    print(f"\n[DONE] Resultats ecrits dans {out_dir}", file=sys.stderr)
    # stdout = JSON pur (pour json.loads() dans les tests et pipelines)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

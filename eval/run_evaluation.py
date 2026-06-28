from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.inference import toy_predict, groq_predict_baseline, groq_predict_improved
from src.guardrails import apply_safety_guardrails, validate_prediction
from src.metrics import summarize_metrics
from src.database import insert_run, insert_evaluation, init_db
from eval.reporting import generate_evaluation_report

# Mapping mode → fonction d'inférence
INFERENCE_MAP = {
    "baseline": lambda p: toy_predict(p, mode="baseline"),
    "improved": lambda p: toy_predict(p, mode="improved"),
    "groq-baseline": groq_predict_baseline,
    "groq-improved": groq_predict_improved,
}


def read_cases(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def classify_error(label: str, predicted: str, image_quality: str, json_valid: bool) -> tuple[str, str, str]:
    """Taxonomie simple pour le registre d'erreurs du rendu."""
    if not json_valid:
        return "JF", "high", "JSON invalide ou sortie inexploitable"
    if label == predicted:
        if predicted == "uncertain":
            return "UA", "low", "Incertitude attendue ou acceptable"
        return "OK", "low", "Prédiction conforme au label de référence"
    if predicted == "uncertain":
        if image_quality in {"limited", "poor"}:
            return "UA", "low", "Abstention prudente liée à une qualité image limitée"
        if label == "suspected_opacity":
            return "FN", "high", "Cas positif non détecté : le modèle s'abstient au lieu de signaler l'opacité"
        return "FP", "medium", "Cas normal non reconnu : abstention excessive"
    if label == "suspected_opacity" and predicted != "suspected_opacity":
        return "FN", "high", "Opacité de référence manquée"
    if label == "normal" and predicted != "normal":
        return "FP", "medium", "Structure normale sur-interprétée"
    if label == "uncertain" and predicted != "uncertain":
        return "HT", "medium", "Décision trop affirmative sur un cas ambigu"
    return "OTHER", "medium", "Erreur à revoir manuellement"


def corrective_action(error_type: str) -> str:
    return {
        "OK": "conserver comme exemple de réussite",
        "UA": "documenter comme incertitude acceptable",
        "FN": "renforcer sensibilité ou revoir seuil d'incertitude",
        "FP": "renforcer spécificité et pénaliser le sur-diagnostic",
        "JF": "durcir le prompt JSON et le parseur",
        "HT": "limiter les conclusions non justifiées par l'image",
    }.get(error_type, "analyse manuelle")


def run(mode: str, db_path: Path, cases_csv: Path | None = None) -> tuple[list[dict], list[dict], dict]:
    """Exécute l'évaluation complète sur un jeu de cas."""
    if cases_csv is None:
        cases_csv = ROOT / "data" / "synthetic" / "cases.csv"
    cases = read_cases(cases_csv)
    predict_fn = INFERENCE_MAP[mode]
    rows: list[dict] = []
    error_rows: list[dict] = []
    init_db(db_path)

    print(f"\n[>>] Evaluation mode '{mode}' sur {len(cases)} cas...", file=sys.stderr)
    for i, case in enumerate(cases, 1):
        image_path = ROOT / case["image_path"]
        pred = apply_safety_guardrails(predict_fn(image_path))
        valid, errors = validate_prediction(pred)
        error_type, severity, comment = classify_error(
            case["label"], pred["predicted_class"], pred.get("image_quality", ""), valid
        )
        row = {
            "case_id": case["case_id"],
            "label": case["label"],
            "predicted_class": pred["predicted_class"],
            "confidence": pred["confidence"],
            "image_quality": pred.get("image_quality", ""),
            "json_valid": valid,
            "warning": pred.get("warning", ""),
            "latency_ms": pred.get("latency_ms", 0),
            "guardrail_errors": ";".join(errors),
            "model_name": pred.get("model_name", mode),
            "prompt_version": pred.get("prompt_version", mode),
            "error_type": error_type,
        }
        rows.append(row)
        run_id = insert_run(db_path, case["case_id"], str(image_path), pred)
        insert_evaluation(
            db_path,
            run_id,
            case["label"],
            case["label"] == pred["predicted_class"],
            error_type,
            comment,
        )
        error_rows.append(
            {
                "case_id": case["case_id"],
                "ground_truth": case["label"],
                "prediction": pred["predicted_class"],
                "error_type": error_type,
                "severity": severity,
                "comment": comment,
                "corrective_action": corrective_action(error_type),
            }
        )

        match = "[OK]" if row["label"] == row["predicted_class"] else "[ERR]"
        print(
            f"  [{i:02d}/{len(cases)}] {match} {case['case_id']:30s} | "
            f"label={row['label']:20s} | pred={row['predicted_class']:20s} | "
            f"conf={row['confidence']:.2f} | {row['latency_ms']}ms",
            file=sys.stderr,
        )

    metrics = summarize_metrics(rows)
    return rows, error_rows, metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Évaluation du prototype radiologue virtuel sur un CSV de cas."
    )
    parser.add_argument(
        "--mode",
        choices=["toy", "baseline", "improved", "groq-baseline", "groq-improved", "all-groq"],
        default="baseline",
        help=(
            "'baseline'/'improved' : toy deterministe. "
            "'groq-baseline'/'groq-improved' : appels API Groq. "
            "'all-groq' : compare les deux modes Groq. "
            "'toy' : alias pour baseline+improved (toy)."
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "eval" / "outputs")
    parser.add_argument("--db-path", type=Path, default=ROOT / "medical_ai_evidence.sqlite")
    parser.add_argument(
        "--cases-csv",
        type=Path,
        default=None,
        help=(
            "CSV de cas à évaluer. "
            "Défaut : data/synthetic/cases.csv. "
            "Utiliser data/chexpert_eval/cases.csv pour les vraies radios CheXpert."
        ),
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "toy":
        modes = ["baseline", "improved"]
    elif args.mode == "all-groq":
        modes = ["groq-baseline", "groq-improved"]
    else:
        modes = [args.mode]

    summary = []
    for mode in modes:
        rows, error_rows, metrics = run(mode, args.db_path, args.cases_csv)
        out_suffix = f"_{args.cases_csv.stem}" if args.cases_csv else ""
        write_csv(out_dir / f"{mode}{out_suffix}_predictions.csv", rows)
        write_csv(out_dir / f"{mode}{out_suffix}_error_register.csv", error_rows)
        (out_dir / f"{mode}{out_suffix}_metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        summary.append({"mode": mode, **metrics})
        print(f"\n[>>] Metriques {mode} :", file=sys.stderr)
        for k, v in metrics.items():
            print(f"   {k}: {v}", file=sys.stderr)

    write_csv(out_dir / "before_after_summary.csv", summary)
    report_path = generate_evaluation_report(out_dir, out_dir / "evaluation_report.md")
    print(f"\n[DONE] Resultats ecrits dans {out_dir}", file=sys.stderr)
    print(f"[DONE] Rapport Markdown : {report_path}", file=sys.stderr)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

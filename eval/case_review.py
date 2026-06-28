from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

ERROR_PRIORITY = {"FN": 0, "FP": 1, "HT": 2, "JF": 3, "UA": 4, "OTHER": 5, "OK": 6}
REVIEW_COLUMNS = [
    "review_rank",
    "mode",
    "case_id",
    "image_path",
    "ground_truth",
    "prediction",
    "confidence",
    "image_quality",
    "error_type",
    "severity",
    "priority_reason",
    "initial_comment",
    "corrective_action",
    "visible_findings_to_describe",
    "human_review_comment",
    "final_decision",
    "screenshot_or_figure_ref",
    "review_status",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] = REVIEW_COLUMNS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _mode_from_prediction_file(path: Path) -> str:
    return path.name.removesuffix("_predictions.csv")


def _float_value(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _priority_reason(error_type: str, severity: str, correct: bool) -> str:
    if error_type == "FN":
        return "Priorité haute : faux négatif potentiel à expliquer et corriger."
    if error_type == "FP":
        return "Priorité moyenne : sur-interprétation ou faux positif à documenter."
    if error_type == "HT":
        return "Conclusion trop affirmative sur un cas ambigu : vérifier les garde-fous."
    if error_type == "JF":
        return "Erreur de format JSON : vérifier prompt, parsing et validation."
    if error_type == "UA":
        return "Incertitude acceptable : utile pour justifier la stratégie prudente."
    if correct:
        return "Cas correct : conserver comme exemple de réussite ou cas témoin."
    if severity:
        return f"Cas à revoir manuellement, sévérité déclarée : {severity}."
    return "Cas à revoir manuellement."


def collect_review_candidates(out_dir: Path) -> list[dict[str, Any]]:
    """Fusionne prédictions et registres d'erreurs en candidats pour revue humaine."""
    out_dir = Path(out_dir)
    candidates: list[dict[str, Any]] = []

    for prediction_file in sorted(out_dir.glob("*_predictions.csv")):
        mode = _mode_from_prediction_file(prediction_file)
        prediction_rows = read_csv(prediction_file)
        error_file = out_dir / prediction_file.name.replace("_predictions.csv", "_error_register.csv")
        error_rows = {row.get("case_id", ""): row for row in read_csv(error_file)}

        for pred in prediction_rows:
            case_id = pred.get("case_id", "")
            err = error_rows.get(case_id, {})
            truth = pred.get("label") or err.get("ground_truth", "")
            prediction = pred.get("predicted_class") or err.get("prediction", "")
            error_type = err.get("error_type") or pred.get("error_type") or "OTHER"
            severity = err.get("severity", "")
            correct = truth == prediction

            candidates.append(
                {
                    "mode": mode,
                    "case_id": case_id,
                    "image_path": pred.get("image_path", ""),
                    "ground_truth": truth,
                    "prediction": prediction,
                    "confidence": pred.get("confidence", ""),
                    "image_quality": pred.get("image_quality", ""),
                    "error_type": error_type,
                    "severity": severity,
                    "priority_reason": _priority_reason(error_type, severity, correct),
                    "initial_comment": err.get("comment", ""),
                    "corrective_action": err.get("corrective_action", ""),
                    "visible_findings_to_describe": "",
                    "human_review_comment": "",
                    "final_decision": "",
                    "screenshot_or_figure_ref": "",
                    "review_status": "à compléter",
                    "_correct": correct,
                    "_confidence_sort": _float_value(pred.get("confidence")),
                }
            )

    return candidates


def _ordered_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        candidates,
        key=lambda row: (
            ERROR_PRIORITY.get(str(row.get("error_type", "OTHER")), 9),
            row.get("_correct", False),
            -float(row.get("_confidence_sort", 0.0)),
            str(row.get("mode", "")),
            str(row.get("case_id", "")),
        ),
    )


def _append_unique(target: list[dict[str, Any]], rows: list[dict[str, Any]], limit: int) -> None:
    seen = {(row.get("mode"), row.get("case_id"), row.get("error_type")) for row in target}
    for row in rows:
        key = (row.get("mode"), row.get("case_id"), row.get("error_type"))
        if key in seen:
            continue
        target.append(row)
        seen.add(key)
        if len(target) >= limit:
            break


def select_review_cases(candidates: list[dict[str, Any]], limit: int = 30) -> list[dict[str, Any]]:
    """Sélectionne les cas les plus utiles à commenter pour le rapport final.

    La sélection priorise les erreurs dangereuses, mais garde aussi des exemples
    corrects. Cela évite un rapport composé uniquement d'incertitudes acceptables
    lorsque le modèle amélioré s'abstient beaucoup.
    """
    limit = max(0, limit)
    ordered = _ordered_candidates(candidates)
    selected: list[dict[str, Any]] = []

    critical = [row for row in ordered if row.get("error_type") in {"FN", "FP", "HT", "JF"}]
    uncertain = [row for row in ordered if row.get("error_type") == "UA"]
    other = [row for row in ordered if row.get("error_type") == "OTHER"]
    correct = [row for row in ordered if row.get("error_type") == "OK"]

    _append_unique(selected, critical, limit)

    # Garder de la place pour des cas corrects/témoins si le jeu d'évaluation en contient.
    witness_target = min(len(correct), max(3, limit // 3)) if correct else 0
    uncertain_limit = max(0, limit - len(selected) - witness_target)
    _append_unique(selected, uncertain[:uncertain_limit], limit)
    _append_unique(selected, other, limit)
    _append_unique(selected, correct, limit)

    # Si les quotas précédents ne suffisent pas, compléter avec les candidats restants.
    if len(selected) < limit:
        _append_unique(selected, ordered, limit)

    public_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(selected[:limit], 1):
        public_rows.append({column: row.get(column, "") for column in REVIEW_COLUMNS})
        public_rows[-1]["review_rank"] = rank
    return public_rows


def build_case_review_template(
    out_dir: Path,
    output_path: Path | None = None,
    limit: int = 30,
) -> Path | None:
    """Crée un CSV remplissable pour les 20-30 cas commentés du rapport final.

    Le fichier généré ne remplace pas l'analyse humaine : il pré-remplit les preuves
    techniques et laisse des colonnes vides à compléter avec captures, observations
    visibles et décisions correctives.
    """
    out_dir = Path(out_dir)
    if output_path is None:
        output_path = out_dir / "case_review_template.csv"

    candidates = collect_review_candidates(out_dir)
    if not candidates:
        return None

    rows = select_review_cases(candidates, limit=limit)
    write_csv(output_path, rows)
    return output_path

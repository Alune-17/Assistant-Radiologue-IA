from __future__ import annotations

import csv
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

CLASSES = ["normal", "suspected_opacity", "uncertain"]
ERROR_LABELS = {
    "OK": "prédiction correcte",
    "UA": "incertitude acceptable",
    "FN": "faux négatif",
    "FP": "faux positif",
    "JF": "erreur JSON / format",
    "HT": "hallucination textuelle ou conclusion trop affirmative",
    "OTHER": "autre erreur à revoir",
}
ERROR_PRIORITY = {"FN": 0, "FP": 1, "HT": 2, "JF": 3, "UA": 4, "OTHER": 5, "OK": 6}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_Aucune donnée disponible._\n"
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        values = [str(row.get(col, "")).replace("\n", " ").replace("|", "\\|") for col in columns]
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *body]) + "\n"


def confusion_matrix_rows(prediction_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in prediction_rows:
        counts[row.get("label", "")][row.get("predicted_class", "")] += 1

    matrix_rows: list[dict[str, Any]] = []
    for label in CLASSES:
        matrix_rows.append(
            {
                "label": label,
                "pred_normal": counts[label]["normal"],
                "pred_suspected_opacity": counts[label]["suspected_opacity"],
                "pred_uncertain": counts[label]["uncertain"],
            }
        )
    return matrix_rows


def summarize_error_register(error_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    counts = Counter(row.get("error_type", "OTHER") for row in error_rows)
    return [
        {"error_type": code, "meaning": ERROR_LABELS.get(code, "à revoir"), "count": count}
        for code, count in counts.most_common()
    ]


def selected_case_rows(error_rows: list[dict[str, str]], limit: int = 30) -> list[dict[str, str]]:
    rows = sorted(
        error_rows,
        key=lambda row: (ERROR_PRIORITY.get(row.get("error_type", "OTHER"), 9), row.get("case_id", "")),
    )
    selected = rows[:limit]
    return [
        {
            "case_id": row.get("case_id", ""),
            "truth": row.get("ground_truth", ""),
            "prediction": row.get("prediction", ""),
            "type": row.get("error_type", ""),
            "commentaire": row.get("comment", ""),
            "action": row.get("corrective_action", ""),
        }
        for row in selected
    ]


def summarize_quality_rows(prediction_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Résume les contrôles qualité image présents dans les prédictions."""
    quality_counts = Counter(row.get("image_quality", "unknown") for row in prediction_rows)
    reason_counts: Counter[str] = Counter()
    for row in prediction_rows:
        for reason in row.get("quality_reasons", "").split(";"):
            reason = reason.strip()
            if reason:
                reason_counts[reason] += 1

    rows = [
        {"type": "quality", "value": quality, "count": count}
        for quality, count in quality_counts.most_common()
    ]
    rows.extend(
        {"type": "reason", "value": reason, "count": count}
        for reason, count in reason_counts.most_common(5)
    )
    return rows


def generate_evaluation_report(out_dir: Path, report_path: Path) -> Path:
    """Génère un rapport Markdown lisible à partir des CSV d'évaluation.

    Le rapport est volontairement factuel : il ne prétend pas valider un modèle médical,
    il rassemble les preuves techniques attendues pour la soutenance.
    """
    out_dir = Path(out_dir)
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    summary_rows = read_csv(out_dir / "before_after_summary.csv")
    prediction_files = sorted(out_dir.glob("*_predictions.csv"))
    error_files = sorted(out_dir.glob("*_error_register.csv"))
    case_review_rows = read_csv(out_dir / "case_review_template.csv")

    lines: list[str] = []
    lines.append("# Rapport d'évaluation automatique")
    lines.append("")
    lines.append(
        f"_Généré le {datetime.now().strftime('%Y-%m-%d %H:%M')} à partir de `{out_dir}`._"
    )
    lines.append("")
    lines.append(
        "> Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise."
    )
    lines.append("")

    lines.append("## 1. Synthèse des métriques")
    lines.append("")
    lines.append(
        markdown_table(
            summary_rows,
            [
                "mode",
                "n",
                "accuracy",
                "macro_f1",
                "sensitivity",
                "specificity",
                "uncertain_rate",
                "json_valid_rate",
                "warning_rate",
                "median_latency_ms",
            ],
        )
    )

    if prediction_files:
        lines.append("## 2. Contrôle qualité image")
        lines.append("")
        lines.append(
            "Cette section documente le prétraitement minimal : résolution, contraste, "
            "luminosité, ratio d'aspect et raisons du flag qualité. Ces contrôles ne valident "
            "pas médicalement l'image ; ils justifient seulement les garde-fous du prototype."
        )
        lines.append("")
        for path in prediction_files:
            rows = read_csv(path)
            lines.append(f"### `{path.name}`")
            lines.append("")
            lines.append(markdown_table(summarize_quality_rows(rows), ["type", "value", "count"]))
            lines.append("")

        lines.append("## 3. Matrices de confusion")
        lines.append("")
        for path in prediction_files:
            rows = read_csv(path)
            lines.append(f"### `{path.name}`")
            lines.append("")
            lines.append(
                markdown_table(
                    confusion_matrix_rows(rows),
                    ["label", "pred_normal", "pred_suspected_opacity", "pred_uncertain"],
                )
            )
            lines.append("")

    if error_files:
        lines.append("## 4. Registres d'erreurs")
        lines.append("")
        for path in error_files:
            rows = read_csv(path)
            lines.append(f"### `{path.name}`")
            lines.append("")
            lines.append(markdown_table(summarize_error_register(rows), ["error_type", "meaning", "count"]))
            lines.append("")
            lines.append("#### Cas à commenter en priorité")
            lines.append("")
            lines.append(
                markdown_table(
                    selected_case_rows(rows, limit=30),
                    ["case_id", "truth", "prediction", "type", "commentaire", "action"],
                )
            )
            lines.append("")

    if case_review_rows:
        lines.append("## 5. Template des 20 à 30 cas commentés")
        lines.append("")
        lines.append(
            "Le fichier `case_review_template.csv` pré-sélectionne les cas à commenter "
            "en priorité pour le rapport final. Les colonnes `visible_findings_to_describe`, "
            "`human_review_comment`, `final_decision` et `screenshot_or_figure_ref` sont à compléter manuellement."
        )
        lines.append("")
        lines.append(
            markdown_table(
                case_review_rows[:10],
                [
                    "review_rank",
                    "mode",
                    "case_id",
                    "ground_truth",
                    "prediction",
                    "error_type",
                    "priority_reason",
                ],
            )
        )
        lines.append("")

    lines.append("## 6. Lecture responsable")
    lines.append("")
    lines.append(
        "Ce rapport sert à défendre la chaîne d'ingénierie : JSON valide, logs, métriques, "
        "incertitude et erreurs documentées. Il ne constitue pas une validation clinique."
    )
    lines.append("")
    lines.append(
        "Pour la soutenance, compléter manuellement 20 à 30 cas avec captures, observations visibles, "
        "erreurs possibles et décision corrective."
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

from src.metrics import summarize_metrics

DEFAULT_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "oui"}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def mode_from_prediction_file(path: Path) -> str:
    """Extrait un nom de mode lisible depuis `baseline_predictions.csv` ou variantes."""
    name = path.name
    suffix = "_predictions.csv"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return path.stem


def apply_uncertainty_threshold(rows: list[dict[str, str]], threshold: float) -> list[dict[str, Any]]:
    """Applique une politique d'abstention par seuil de confiance.

    Règle pédagogique : si la confiance est inférieure au seuil, ou si la qualité
    image est `poor`, la prédiction est remplacée par `uncertain`. On ne modifie
    pas les autres champs : l'objectif est de comparer des politiques de sécurité
    à modèle constant.
    """
    adjusted: list[dict[str, Any]] = []
    for row in rows:
        original = row.get("predicted_class", "uncertain")
        confidence = _as_float(row.get("confidence"), 0.0)
        image_quality = row.get("image_quality", "")
        new_row: dict[str, Any] = dict(row)
        new_row["label"] = row.get("label", "")
        new_row["json_valid"] = _as_bool(row.get("json_valid", True))
        new_row["warning"] = row.get("warning", "")
        new_row["latency_ms"] = _as_float(row.get("latency_ms"), 0.0)
        new_row["predicted_class"] = original
        new_row["threshold_changed_to_uncertain"] = False

        if image_quality == "poor" or confidence < threshold:
            new_row["predicted_class"] = "uncertain"
            new_row["threshold_changed_to_uncertain"] = original != "uncertain"

        adjusted.append(new_row)
    return adjusted


def sweep_prediction_file(path: Path, thresholds: list[float] | None = None) -> list[dict[str, Any]]:
    """Calcule les métriques obtenues pour plusieurs seuils sur un fichier de prédictions."""
    rows = read_csv(path)
    if not rows:
        return []
    thresholds = thresholds or DEFAULT_THRESHOLDS
    mode = mode_from_prediction_file(path)

    sweep_rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        adjusted = apply_uncertainty_threshold(rows, threshold)
        metrics = summarize_metrics(adjusted)
        changed_count = sum(bool(row.get("threshold_changed_to_uncertain")) for row in adjusted)
        sweep_rows.append(
            {
                "prediction_file": path.name,
                "mode": mode,
                "threshold": round(float(threshold), 2),
                **metrics,
                "changed_to_uncertain_rate": round(changed_count / len(adjusted), 4) if adjusted else 0.0,
            }
        )
    return sweep_rows


def generate_threshold_sweep(out_dir: Path, thresholds: list[float] | None = None) -> Path | None:
    """Génère `threshold_sweep.csv` à partir des `*_predictions.csv` existants."""
    out_dir = Path(out_dir)
    prediction_files = sorted(out_dir.glob("*_predictions.csv"))
    if not prediction_files:
        return None

    rows: list[dict[str, Any]] = []
    for path in prediction_files:
        rows.extend(sweep_prediction_file(path, thresholds=thresholds))

    if not rows:
        return None

    output_path = out_dir / "threshold_sweep.csv"
    write_csv(output_path, rows)
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyse des seuils d'incertitude à partir des prédictions CSV.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "eval" / "outputs")
    parser.add_argument(
        "--thresholds",
        default="0.50,0.55,0.60,0.65,0.70,0.75,0.80",
        help="Liste de seuils séparés par des virgules.",
    )
    args = parser.parse_args()
    thresholds = [float(value.strip()) for value in args.thresholds.split(",") if value.strip()]
    path = generate_threshold_sweep(args.out_dir, thresholds=thresholds)
    if path is None:
        raise SystemExit("Aucun fichier *_predictions.csv trouvé.")
    print(path)

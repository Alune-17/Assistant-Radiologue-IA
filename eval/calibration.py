from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BINS = [0.0, 0.20, 0.40, 0.60, 0.80, 1.000001]


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


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "oui"}


def mode_from_prediction_file(path: Path) -> str:
    suffix = "_predictions.csv"
    if path.name.endswith(suffix):
        return path.name[: -len(suffix)]
    return path.stem


def _bin_label(low: float, high: float) -> str:
    shown_high = min(high, 1.0)
    return f"[{low:.2f};{shown_high:.2f}]"


def _reliability_flag(avg_confidence: float, accuracy: float, n: int) -> str:
    if n == 0:
        return "empty"
    gap = avg_confidence - accuracy
    if abs(gap) <= 0.10:
        return "aligned"
    if gap > 0:
        return "overconfident"
    return "underconfident"


def calibration_rows_for_predictions(
    rows: list[dict[str, str]],
    mode: str,
    prediction_file: str,
    bins: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Calcule une table de calibration par intervalles de confiance.

    La calibration compare la confiance moyenne du modèle et la proportion réelle
    de prédictions correctes dans chaque intervalle. C'est une preuve utile pour
    discuter la fiabilité du score de confiance sans prétendre à une validation clinique.
    """
    bins = bins or DEFAULT_BINS
    if len(bins) < 2:
        raise ValueError("La calibration nécessite au moins deux bornes de bins.")

    parsed_rows: list[dict[str, Any]] = []
    for row in rows:
        confidence = max(0.0, min(1.0, _as_float(row.get("confidence"), 0.0)))
        label = row.get("label", "")
        predicted = row.get("predicted_class", "")
        parsed_rows.append(
            {
                "confidence": confidence,
                "correct": label == predicted,
                "is_uncertain": predicted == "uncertain",
                "json_valid": _as_bool(row.get("json_valid", True)),
            }
        )

    total = len(parsed_rows)
    output_rows: list[dict[str, Any]] = []
    ece = 0.0

    for low, high in zip(bins[:-1], bins[1:]):
        in_bin = [row for row in parsed_rows if low <= row["confidence"] < high]
        n = len(in_bin)
        avg_confidence = sum(row["confidence"] for row in in_bin) / n if n else 0.0
        accuracy = sum(bool(row["correct"]) for row in in_bin) / n if n else 0.0
        uncertain_rate = sum(bool(row["is_uncertain"]) for row in in_bin) / n if n else 0.0
        json_valid_rate = sum(bool(row["json_valid"]) for row in in_bin) / n if n else 0.0
        abs_gap = abs(avg_confidence - accuracy) if n else 0.0
        if total:
            ece += (n / total) * abs_gap
        output_rows.append(
            {
                "mode": mode,
                "prediction_file": prediction_file,
                "row_type": "bin",
                "confidence_bin": _bin_label(low, high),
                "n": n,
                "coverage": round(n / total, 4) if total else 0.0,
                "avg_confidence": round(avg_confidence, 4),
                "accuracy": round(accuracy, 4),
                "calibration_gap": round(avg_confidence - accuracy, 4) if n else 0.0,
                "abs_gap": round(abs_gap, 4),
                "uncertain_rate": round(uncertain_rate, 4),
                "json_valid_rate": round(json_valid_rate, 4),
                "ece": "",
                "reliability_flag": _reliability_flag(avg_confidence, accuracy, n),
            }
        )

    overall_accuracy = sum(bool(row["correct"]) for row in parsed_rows) / total if total else 0.0
    overall_confidence = sum(row["confidence"] for row in parsed_rows) / total if total else 0.0
    overall_uncertain_rate = sum(bool(row["is_uncertain"]) for row in parsed_rows) / total if total else 0.0
    overall_json_valid_rate = sum(bool(row["json_valid"]) for row in parsed_rows) / total if total else 0.0
    output_rows.insert(
        0,
        {
            "mode": mode,
            "prediction_file": prediction_file,
            "row_type": "overall",
            "confidence_bin": "all",
            "n": total,
            "coverage": 1.0 if total else 0.0,
            "avg_confidence": round(overall_confidence, 4),
            "accuracy": round(overall_accuracy, 4),
            "calibration_gap": round(overall_confidence - overall_accuracy, 4) if total else 0.0,
            "abs_gap": round(abs(overall_confidence - overall_accuracy), 4) if total else 0.0,
            "uncertain_rate": round(overall_uncertain_rate, 4),
            "json_valid_rate": round(overall_json_valid_rate, 4),
            "ece": round(ece, 4),
            "reliability_flag": _reliability_flag(overall_confidence, overall_accuracy, total),
        },
    )
    return output_rows


def generate_calibration_report(out_dir: Path, bins: list[float] | None = None) -> Path | None:
    """Génère `calibration_report.csv` à partir des fichiers `*_predictions.csv`."""
    out_dir = Path(out_dir)
    prediction_files = sorted(out_dir.glob("*_predictions.csv"))
    if not prediction_files:
        return None

    rows: list[dict[str, Any]] = []
    for path in prediction_files:
        prediction_rows = read_csv(path)
        if not prediction_rows:
            continue
        rows.extend(
            calibration_rows_for_predictions(
                prediction_rows,
                mode=mode_from_prediction_file(path),
                prediction_file=path.name,
                bins=bins,
            )
        )

    if not rows:
        return None

    output_path = out_dir / "calibration_report.csv"
    write_csv(output_path, rows)
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyse de calibration des scores de confiance.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "eval" / "outputs")
    parser.add_argument(
        "--bins",
        default="0.0,0.2,0.4,0.6,0.8,1.0",
        help="Bornes des intervalles de confiance, séparées par des virgules.",
    )
    args = parser.parse_args()
    bins = [float(value.strip()) for value in args.bins.split(",") if value.strip()]
    if bins and bins[-1] == 1.0:
        bins[-1] = 1.000001
    path = generate_calibration_report(args.out_dir, bins=bins)
    if path is None:
        raise SystemExit("Aucun fichier *_predictions.csv trouvé.")
    print(path)

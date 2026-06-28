from __future__ import annotations

from typing import Any

ALLOWED_CLASSES = {"normal", "suspected_opacity", "uncertain"}
ALLOWED_QUALITY = {"good", "limited", "poor"}
REQUIRED_KEYS = {"image_quality", "predicted_class", "confidence", "visual_evidence", "justification", "limitations", "warning"}
WARNING_TEXT = "Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise."
UNCERTAINTY_THRESHOLD = 0.60


def validate_prediction(pred: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    missing = REQUIRED_KEYS - set(pred)
    if missing:
        errors.append(f"missing keys: {sorted(missing)}")
    if pred.get("image_quality") not in ALLOWED_QUALITY:
        errors.append("invalid image_quality")
    if pred.get("predicted_class") not in ALLOWED_CLASSES:
        errors.append("invalid predicted_class")
    try:
        conf = float(pred.get("confidence", -1))
        if not 0 <= conf <= 1:
            errors.append("confidence outside [0,1]")
    except Exception:
        errors.append("confidence is not numeric")
    if not isinstance(pred.get("visual_evidence", []), list):
        errors.append("visual_evidence is not a list")
    if not isinstance(pred.get("limitations", []), list):
        errors.append("limitations is not a list")
    if not pred.get("warning"):
        errors.append("warning missing")
    return not errors, errors


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def apply_safety_guardrails(pred: dict[str, Any]) -> dict[str, Any]:
    """Applique le contrat de sécurité du projet.

    Toute sortie invalide ou trop peu fiable est ramenée vers `uncertain`.
    Le warning non clinique est injecté systématiquement.
    """
    pred = dict(pred)
    original_valid, errors = validate_prediction(pred)

    pred["image_quality"] = pred.get("image_quality") if pred.get("image_quality") in ALLOWED_QUALITY else "poor"
    pred["visual_evidence"] = _as_list(pred.get("visual_evidence"))
    pred["limitations"] = _as_list(pred.get("limitations"))
    pred["justification"] = str(pred.get("justification", "Sortie incomplète ou invalide."))
    pred["confidence"] = max(0.0, min(1.0, _as_float(pred.get("confidence"), 0.0)))

    if pred.get("predicted_class") not in ALLOWED_CLASSES:
        pred["predicted_class"] = "uncertain"

    if not original_valid:
        pred["predicted_class"] = "uncertain"
        pred["confidence"] = min(pred["confidence"], 0.5)
        pred["limitations"].append("guardrail triggered: invalid output schema")

    # Règle centrale du sujet : pas de décision forcée si la qualité est pauvre,
    # ou si la qualité est limitée avec une confiance insuffisante.
    if pred["image_quality"] == "poor" or (
        pred["image_quality"] == "limited" and pred["confidence"] < UNCERTAINTY_THRESHOLD
    ):
        pred["predicted_class"] = "uncertain"
        pred["confidence"] = min(pred["confidence"], UNCERTAINTY_THRESHOLD)

    pred["warning"] = WARNING_TEXT
    pred["guardrail_errors"] = errors
    return pred

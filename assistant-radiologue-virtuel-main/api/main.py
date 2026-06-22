from __future__ import annotations

import re
import shutil
from pathlib import Path
from fastapi import FastAPI, File, Query, UploadFile

from src.inference import toy_predict, gemini_predict_baseline, gemini_predict_improved
from src.guardrails import apply_safety_guardrails

app = FastAPI(
    title="Assistant radiologue virtuel EFREI",
    version="0.2.0",
    description="Prototype pédagogique — non destiné au diagnostic médical.",
)
UPLOAD_DIR = Path("tmp_uploads")

# Mapping nom → fonction d'inférence
MODELS = {
    "toy":              lambda p: toy_predict(p, mode="baseline"),
    "gemini-baseline":  gemini_predict_baseline,
    "gemini-improved":  gemini_predict_improved,
}


@app.get("/")
def health() -> dict:
    return {
        "status": "ok",
        "scope": "educational prototype, not diagnosis",
        "available_models": list(MODELS.keys()),
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model: str = Query(
        default="toy",
        description="Modèle d'inférence : 'toy', 'gemini-baseline' ou 'gemini-improved'.",
    ),
) -> dict:
    """Analyse une radiographie thoracique frontale et retourne un JSON structuré.

    - **file**: Image PNG/JPG/JPEG/BMP à analyser.
    - **model**: Sélection du modèle (`toy` par défaut, pas d'appel API externe).
    """
    UPLOAD_DIR.mkdir(exist_ok=True)
    filename = Path(file.filename or "image.png").name
    suffix = Path(filename).suffix or ".png"
    stem = Path(filename).stem or "image"
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
    target = UPLOAD_DIR / f"uploaded_{safe_stem}{suffix}"
    with target.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    predict_fn = MODELS.get(model, MODELS["toy"])
    pred = predict_fn(target)
    return apply_safety_guardrails(pred)

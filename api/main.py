from __future__ import annotations

import os
import re
from pathlib import Path
from uuid import uuid4
from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from src.inference import toy_predict, groq_predict_baseline, groq_predict_improved
from src.guardrails import apply_safety_guardrails
from src.preprocessing import validate_image_upload
from src.database import insert_run

app = FastAPI(
    title="Assistant radiologue virtuel EFREI",
    version="0.5.0",
    description="Prototype pédagogique — non destiné au diagnostic médical.",
)
UPLOAD_DIR = Path("tmp_uploads")
DB_PATH = Path(os.getenv("ARVIRX_DB_PATH", str(UPLOAD_DIR / "medical_ai_evidence.sqlite")))

# Mapping nom → fonction d'inférence
MODELS = {
    "toy": lambda p: toy_predict(p, mode="baseline"),
    "groq-baseline": groq_predict_baseline,
    "groq-improved": groq_predict_improved,
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
        description="Modèle d'inférence : 'toy', 'groq-baseline' ou 'groq-improved'.",
    ),
) -> dict:
    """Analyse une radiographie thoracique frontale et retourne un JSON structuré.

    - **file**: Image PNG/JPG/JPEG/BMP à analyser.
    - **model**: Sélection du modèle (`toy` par défaut, pas d'appel API externe).
    """
    if model not in MODELS:
        raise HTTPException(status_code=400, detail=f"Modèle inconnu. Valeurs autorisées : {sorted(MODELS)}")

    UPLOAD_DIR.mkdir(exist_ok=True)
    filename = Path(file.filename or "image.png").name
    content = await file.read()
    try:
        suffix = validate_image_upload(filename, content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    stem = Path(filename).stem or "image"
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)[:80]
    target = UPLOAD_DIR / f"uploaded_{safe_stem}_{uuid4().hex[:8]}{suffix}"
    target.write_bytes(content)

    pred = apply_safety_guardrails(MODELS[model](target))
    try:
        insert_run(DB_PATH, f"upload:{safe_stem}", str(target), pred)
    except Exception as exc:  # Le rendu JSON ne doit pas tomber si SQLite est indisponible.
        pred.setdefault("limitations", []).append(f"log SQLite non enregistré: {type(exc).__name__}")
    return pred

from __future__ import annotations

from pathlib import Path
import time
from typing import Any
import os
import json
import PIL.Image
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Client Gemini initialisé une seule fois (None si la clé est absente)
_client: genai.Client | None = None
if os.getenv("GOOGLE_API_KEY"):
    _client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

from .preprocessing import basic_quality_flag

WARNING = "Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise."

# Modèle choisi : gemini-2.0-flash — meilleur compromis disponibilité/quota/stabilité API
GEMINI_MODEL = "gemini-2.0-flash"

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def toy_predict(image_path: str | Path, mode: str = "baseline") -> dict[str, Any]:
    """Deterministic toy predictor used to validate the repo pipeline.

    It reads synthetic labels from filenames. This is not medical inference.
    """
    start = time.perf_counter()
    name = Path(image_path).name.lower()
    quality = basic_quality_flag(image_path)

    if "suspected_opacity" in name:
        pred = "suspected_opacity"
        conf = 0.78 if mode == "baseline" else 0.72
        evidence = ["synthetic opacity-like area visible in the lung field"]
        justification = "The synthetic image contains a localized brighter region compatible with the toy opacity class. This is a pipeline validation result, not a medical interpretation."
    elif "normal" in name:
        pred = "normal"
        conf = 0.72 if mode == "baseline" else 0.68
        evidence = ["no synthetic opacity marker detected"]
        justification = "The synthetic image does not contain the opacity marker used by the toy generator. This conclusion is limited to the synthetic validation setting."
    else:
        pred = "uncertain"
        conf = 0.52
        evidence = ["limited synthetic image quality"]
        justification = "The image is treated as limited quality in the toy catalog. The safe output is uncertainty rather than a forced class."

    # Improved mode is more conservative.
    if mode == "improved" and quality != "good":
        pred = "uncertain"
        conf = min(conf, 0.55)

    latency_ms = int((time.perf_counter() - start) * 1000)
    return {
        "image_quality": quality,
        "predicted_class": pred,
        "confidence": round(float(conf), 3),
        "visual_evidence": evidence,
        "justification": justification,
        "limitations": ["synthetic toy image", "no clinical context", "not a validated medical model"],
        "warning": WARNING,
        "model_name": f"toy-rule-{mode}",
        "prompt_version": f"{mode}_v1",
        "latency_ms": latency_ms,
    }


def vlm_predict_placeholder(image_path: str | Path, prompt: str) -> dict[str, Any]:
    """Placeholder for a Hugging Face / MedGemma / Gemma 4 VLM call.

    Students should keep the same output schema as toy_predict.
    """
    return toy_predict(image_path, mode="baseline")


def _gemini_call(
    image_path: str | Path,
    prompt_file: str,
    prompt_version: str,
) -> dict[str, Any]:
    """Fonction interne commune pour tous les appels API Gemini.

    Args:
        image_path: Chemin vers l'image à analyser.
        prompt_file: Nom du fichier de prompt dans prompts/ (ex. 'baseline_prompt.txt').
        prompt_version: Identifiant de version pour les logs (ex. 'baseline_v1').

    Returns:
        Dictionnaire JSON conforme au schéma du projet.
    """
    start = time.perf_counter()
    quality = basic_quality_flag(image_path)

    # Structure de réponse par défaut (fallback sécurisé)
    result: dict[str, Any] = {
        "image_quality": quality,
        "predicted_class": "uncertain",
        "confidence": 0.0,
        "visual_evidence": [],
        "justification": "Erreur lors de l'appel API ou format de réponse invalide.",
        "limitations": [
            "Modèle générique non spécialisé en radiologie",
            "Dépendance à une connexion Internet",
            "Images synthétiques simplistes — pas des vraies radiographies",
            "Pas un dispositif médical validé",
        ],
        "warning": WARNING,
        "model_name": GEMINI_MODEL,
        "prompt_version": prompt_version,
        "latency_ms": 0,
    }

    if _client is None:
        result["justification"] = (
            "Clé API GOOGLE_API_KEY manquante dans le fichier .env. "
            "Créez un fichier .env à la racine du projet avec votre clé."
        )
        return result

    try:
        # Chargement du prompt
        prompt_path = PROMPTS_DIR / prompt_file
        with open(prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read()

        img = PIL.Image.open(image_path)

        # Forcer la réponse en JSON pour intégration avec les guardrails
        response = _client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[system_prompt, img],
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            ),
        )

        response_data = json.loads(response.text)

        # Extraction sécurisée — on complète les champs potentiellement manquants
        result["predicted_class"] = response_data.get("predicted_class", "uncertain")
        result["confidence"] = round(float(response_data.get("confidence", 0.0)), 3)
        result["visual_evidence"] = response_data.get("visual_evidence", [])
        result["justification"] = response_data.get("justification", "")

        # image_quality et limitations peuvent être fournis par le modèle
        if "image_quality" in response_data:
            result["image_quality"] = response_data["image_quality"]
        if "limitations" in response_data:
            result["limitations"] = response_data["limitations"]

        # Règle d'incertitude : si confidence < 0.60, forcer uncertain (garde-fou)
        if result["confidence"] < 0.60:
            result["predicted_class"] = "uncertain"

    except json.JSONDecodeError as e:
        result["justification"] = (
            f"La réponse du modèle n'est pas du JSON valide : {str(e)[:120]}. "
            "Classe incertaine par sécurité."
        )
    except Exception as e:
        result["justification"] = f"Erreur API Gemini ({type(e).__name__}) : {str(e)[:200]}"

    result["latency_ms"] = int((time.perf_counter() - start) * 1000)
    return result


def gemini_predict(image_path: str | Path) -> dict[str, Any]:
    """Prédiction via l'API Gemini avec le prompt français générique (gemini_prompt.txt).

    Conservé pour compatibilité avec l'interface Streamlit existante.
    Utilise : prompts/gemini_prompt.txt
    """
    return _gemini_call(image_path, "gemini_prompt.txt", "gemini_prompt_v1")


def gemini_predict_baseline(image_path: str | Path) -> dict[str, Any]:
    """Prédiction Gemini avec le prompt baseline anglais (baseline_prompt.txt).

    Prompt simple : analyse basique sans règles d'incertitude strictes.
    Utilise : prompts/baseline_prompt.txt
    """
    return _gemini_call(image_path, "baseline_prompt.txt", "baseline_v1")


def gemini_predict_improved(image_path: str | Path) -> dict[str, Any]:
    """Prédiction Gemini avec le prompt amélioré (improved_prompt.txt).

    Prompt renforcé : règles d'incertitude strictes, vérification des artefacts.
    Utilise : prompts/improved_prompt.txt
    """
    return _gemini_call(image_path, "improved_prompt.txt", "improved_v1")

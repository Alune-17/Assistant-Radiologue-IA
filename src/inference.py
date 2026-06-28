from __future__ import annotations

from pathlib import Path
import time
from typing import Any
import os
import json
import re
import base64
from dotenv import load_dotenv

try:
    from groq import Groq
except ImportError:  # Permet aux tests et au mode toy de fonctionner sans dépendance API.
    Groq = None  # type: ignore[assignment]

load_dotenv()

# Client Groq initialisé une seule fois (None si la clé ou le package est absent)
_client: Any | None = None
if Groq is not None and os.getenv("GROQ_API_KEY"):
    _client = Groq(api_key=os.environ["GROQ_API_KEY"])

from .preprocessing import image_quality_metadata

WARNING = "Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise."

# Modèle choisi : Llama 4 Scout (modèle Vision officiel en attendant Qwen)
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def toy_predict(image_path: str | Path, mode: str = "baseline") -> dict[str, Any]:
    """Deterministic toy predictor used to validate the repo pipeline.

    It reads synthetic labels from filenames. This is not medical inference.
    """
    start = time.perf_counter()
    name = Path(image_path).name.lower()
    quality_checks = image_quality_metadata(image_path)
    quality = str(quality_checks["quality"])

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
        "quality_checks": quality_checks,
        "model_name": f"toy-rule-{mode}",
        "prompt_version": f"{mode}_v1",
        "latency_ms": latency_ms,
    }


def vlm_predict_placeholder(image_path: str | Path, prompt: str) -> dict[str, Any]:
    """Placeholder for a Hugging Face / MedGemma / Gemma 4 VLM call.

    Students should keep the same output schema as toy_predict.
    """
    return toy_predict(image_path, mode="baseline")


def _encode_image(image_path: str | Path) -> str:
    """Encode une image en base64 pour l'API Groq."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def _groq_call(
    image_path: str | Path,
    prompt_file: str,
    prompt_version: str,
) -> dict[str, Any]:
    """Fonction interne commune pour tous les appels API Groq Vision.

    Args:
        image_path: Chemin vers l'image à analyser.
        prompt_file: Nom du fichier de prompt dans prompts/ (ex. 'baseline_prompt.txt').
        prompt_version: Identifiant de version pour les logs (ex. 'baseline_v1').

    Returns:
        Dictionnaire JSON conforme au schéma du projet.
    """
    start = time.perf_counter()
    quality_checks = image_quality_metadata(image_path)
    quality = str(quality_checks["quality"])

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
        "quality_checks": quality_checks,
        "model_name": GROQ_MODEL,
        "prompt_version": prompt_version,
        "latency_ms": 0,
    }

    if _client is None:
        if Groq is None:
            result["justification"] = (
                "Le package Python 'groq' n'est pas installé. "
                "Installez les dépendances avec `pip install -r requirements.txt` ou utilisez le mode toy local."
            )
        else:
            result["justification"] = (
                "Clé API GROQ_API_KEY manquante dans le fichier .env. "
                "Créez un fichier .env à la racine du projet avec votre clé gratuite (console.groq.com)."
            )
        result["latency_ms"] = int((time.perf_counter() - start) * 1000)
        return result

    try:
        # Chargement du prompt
        prompt_path = PROMPTS_DIR / prompt_file
        with open(prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read()

        base64_image = _encode_image(image_path)

        # Forcer la réponse en JSON pour intégration avec les guardrails
        # Groq Llama Vision supporte le JSON mode nativement
        response = _client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": system_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            temperature=0.1,
        )

        response_content = response.choices[0].message.content
        if not response_content:
            raise ValueError("Réponse vide de Groq")
            
        # Extract JSON using regex to handle markdown wrappers like ```json ... ```
        json_match = re.search(r"\{.*\}", response_content, re.DOTALL)
        if json_match:
            response_data = json.loads(json_match.group(0))
        else:
            response_data = json.loads(response_content)

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
        result["justification"] = f"Erreur API Groq ({type(e).__name__}) : {str(e)[:200]}"

    result["latency_ms"] = int((time.perf_counter() - start) * 1000)
    return result


def groq_predict(image_path: str | Path) -> dict[str, Any]:
    """Prédiction via l'API Groq avec le prompt générique."""
    return _groq_call(image_path, "french_prompt.txt", "vlm_prompt_v1")


def groq_predict_baseline(image_path: str | Path) -> dict[str, Any]:
    """Prédiction Groq avec le prompt baseline anglais (baseline_prompt.txt)."""
    return _groq_call(image_path, "baseline_prompt.txt", "baseline_v1")


def groq_predict_improved(image_path: str | Path) -> dict[str, Any]:
    """Prédiction Groq avec le prompt amélioré (improved_prompt.txt)."""
    return _groq_call(image_path, "improved_prompt.txt", "improved_v1")

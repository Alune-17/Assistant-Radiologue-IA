# Architecture du prototype

> **Projet :** Assistant Radiologue Virtuel — EFREI Solution Delivery  
> **Année académique :** 2025-2026

## Pipeline de traitement

```text
Image upload → preprocessing.py → inference.py → guardrails.py → JSON structuré → UI / API → SQLite logs
                (qualité image)     (toy ou Groq)   (validation)    (résultat)
```

## Modes d'inférence disponibles

| Mode | Fonction | Description |
|---|---|---|
| `toy` | `toy_predict()` | Prédicteur déterministe basé sur les noms de fichiers — sert uniquement à valider le pipeline |
| `groq-baseline` | `groq_predict_baseline()` | API Groq / Llama 4 Scout avec `baseline_prompt.txt` (anglais, minimaliste) |
| `groq-improved` | `groq_predict_improved()` | API Groq / Llama 4 Scout avec `improved_prompt.txt` (anglais, garde-fous renforcés) |
| `groq-fr` | `groq_predict()` | API Groq / Llama 4 Scout avec `french_prompt.txt` (français) |

## Composants source (`src/`)

| Fichier | Rôle |
|---|---|
| `preprocessing.py` | Chargement image, analyse qualité réelle (luminosité, contraste, résolution, ratio d'aspect) |
| `inference.py` | Inférence toy (déterministe) + appels API Groq avec gestion d'erreurs et fallback |
| `guardrails.py` | Validation du schéma JSON, injection du warning obligatoire, fallback `uncertain` si invalide |
| `metrics.py` | Accuracy, macro-F1, sensibilité, spécificité, latence médiane, taux JSON valide / warning |
| `database.py` | Connecteur SQLite pour journaliser chaque run dans la table `runs` |

## Points d'entrée

| Fichier | Technologie | Usage |
|---|---|---|
| `api/main.py` | FastAPI | `GET /` (health) + `POST /predict?model=toy\|groq-baseline\|groq-improved` |
| `app/streamlit_app.py` | Streamlit | Interface web avec upload, sélection de modèle, affichage résultats |
| `app/gradio_app.py` | Gradio | Interface web alternative minimale (mode toy uniquement) |
| `eval/run_evaluation.py` | Script CLI | Évaluation batch sur les cas synthétiques ou CheXpert |

## Schéma JSON de sortie attendu

```json
{
  "image_quality": "good | limited | poor",
  "predicted_class": "normal | suspected_opacity | uncertain",
  "confidence": 0.0,
  "visual_evidence": ["observation 1", "observation 2"],
  "justification": "2 à 4 phrases factuelles",
  "limitations": ["limite 1", "limite 2"],
  "warning": "Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise.",
  "model_name": "toy-rule-baseline | meta-llama/llama-4-scout-17b-16e-instruct",
  "prompt_version": "baseline_v1 | improved_v1 | vlm_prompt_v1",
  "latency_ms": 0
}
```

## Objectifs d'intégration

- ≥ 95% JSON valide
- 100% des sorties avec warning
- 100% des runs sauvegardés en SQLite
- Latence cible < 10 s en mode prototype

## Base de données SQLite

Le schéma (`sql/schema.sql`) définit 4 tables :

| Table | Statut | Rôle |
|---|---|---|
| `cases` | ⚠️ Non peuplée | Index des cas d'évaluation |
| `prompts` | ⚠️ Non peuplée | Historique des prompts utilisés |
| `runs` | ✅ Utilisée | Journalisation de chaque prédiction |
| `evaluations` | ⚠️ Non peuplée | Résultats du registre d'erreurs |

# Changelog — Assistant Radiologue Virtuel Responsable

> Format : [Keep a Changelog](https://keepachangelog.com/fr/1.0.0/)  
> Versioning : [Semantic Versioning](https://semver.org/lang/fr/)

---

## [Unreleased] — À venir

### À faire
- Évaluation complète `groq-baseline` vs `groq-improved` sur 30 radios CheXpert
- Notebook `01_baseline_vlm.ipynb` — documentation résultats
- Registre d'erreurs commenté (20-30 cas)
- Dashboard de métriques Streamlit
- Rapport final

---

## [0.4.0] — 2026-06-27 — Réorganisation du dépôt

### Modifié
- `README.md` — réécrit comme vitrine claire sans doublons avec `docs/`
- `requirements.txt` — séparé en core (minimal) et GPU (optionnel)
- `docs/architecture.md` — mis à jour avec le pipeline réel (Groq / Llama 4 Scout)
- `src/__init__.py` — ajout docstring de package
- `.env.example` — correction typo « Grok » → « Groq »
- `tests/test_repository_smoke.py` — vérifie `docs/appel_offre.pdf` (fichier source)
- `.gitignore` — ajout `cahier_des_charges.md` (document de suivi personnel)

### Renommé
- `prompts/gemini_prompt.txt` → `prompts/french_prompt.txt` (nom hérité de la migration, prêtait à confusion)
- `src/inference.py` — mise à jour de la référence au prompt français

### Déplacé
- `rapport_evaluation.md` → `docs/rapport_evaluation.md` (livrable à sa place avec la doc)

### Supprimé
- `docs/appel_offre.md` — doublon du PDF original `docs/appel_offre.pdf`

### Créé
- `requirements-gpu.txt` — dépendances optionnelles pour fine-tuning (torch, transformers, etc.)

---

## [0.3.0] — 2026-06-22 — Migration Groq (Llama 4 Scout)

### Contexte
L'API Gemini présentait des erreurs de quota/billing en Europe. Le projet a été migré vers
l'API **Groq** (gratuite, rapide) avec le modèle **Llama 4 Scout** (`meta-llama/llama-4-scout-17b-16e-instruct`).

### Modifié
- `src/inference.py` — Remplacement complet du client Gemini par le SDK `groq`
- `app/streamlit_app.py` — Labels d'interface mis à jour pour Groq
- `eval/run_evaluation.py` — Modes `--mode groq-*` au lieu de `gemini-*`
- `api/main.py` — Routage API mis à jour
- `requirements.txt` — `google-genai` remplacé par `groq`
- `.env.example` — `GOOGLE_API_KEY` remplacé par `GROQ_API_KEY`

### Ajouté
- Parsing JSON robuste via `re.search` pour gérer les réponses Markdown de l'API Groq

---

## [0.2.0] — 2026-06-22 — Dataset réel CheXpert

### Ajouté
- `data/download_chexpert.py` — Script de téléchargement via Kaggle API (sous-ensemble de 30 cas)
- `data/README.md` — Documentation complète des datasets (synthétique vs CheXpert)
- Mapping automatique des labels CheXpert → 3 classes du projet

### Modifié
- `eval/run_evaluation.py` — paramètre `--cases-csv` pour évaluer sur CheXpert

---

## [0.1.0] — 2026-06-22 — Infrastructure API et nettoyage initial

### Ajouté
- `.env.example` — Template de configuration
- Fonctions d'inférence API : `_groq_call()`, `groq_predict_baseline()`, `groq_predict_improved()`
- Garde-fou : confiance < 0.60 → classe forcée à `uncertain`

### Modifié
- `src/preprocessing.py` — Analyse de qualité image réelle (luminosité, contraste, résolution)
- `src/metrics.py` — Ajout sensibilité, spécificité, latence médiane
- `app/streamlit_app.py` — Interface 4 modes, 3 colonnes, spinner, affichage latence

### Supprimé
- `medical_ai_evidence.sqlite` — Base SQLite commitée par erreur
- `eval/outputs/` — Résultats commités par erreur

### Corrigé
- `UnicodeEncodeError` sous Windows (emojis → ASCII dans les logs)

---

## [0.0.1] — État initial du repository (fourni par le professeur)

### Hérité
- Architecture de base : `src/`, `api/`, `app/`, `eval/`, `data/`, `tests/`, `docs/`
- `toy_predict()` déterministe + squelettes d'inférence VLM
- Guardrails, database SQLite, CI GitHub Actions
- 30 cas synthétiques (images PNG, pas des radios)
- 7 smoke tests
